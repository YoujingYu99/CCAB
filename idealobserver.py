import os
from functools import partial

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributions as D

from particle import Particle, ParticleJCRP
from spmd_wp import SPMDWP
from subject import Subject
from utils_sample import bern_sample


def pf_worker_loop(
    worker_id: int,
    particle_indices: list[int],
    hyp_params: dict,
    cmd_q: mp.Queue,
    res_q: mp.Queue,
    particle_cls,
):
    """
    Worker loop for particle filtering.

    Each worker owns a slice of the global particle population and
    executes commands on its local particles.

    `particle_cls` is either Particle or ParticleJCRP.
    """

    # Disable within-worker multithreading to prevent oversubscription.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Create local particles for this worker.
    local_particles = [particle_cls(**hyp_params) for _ in particle_indices]

    local_logw = torch.zeros(
        len(local_particles),
        dtype=torch.float64,
    )

    while True:
        msg = cmd_q.get()

        if msg is None:
            break

        cmd = msg[0]

        # ------------------------------------------------------------------
        # BEFORE
        # ------------------------------------------------------------------
        if cmd == "before":
            (o_t,) = msg[1:]

            for particle in local_particles:
                particle.before_action(o_t)

            res_q.put(("before_done", worker_id))

        # ------------------------------------------------------------------
        # ACT
        # ------------------------------------------------------------------
        elif cmd == "act":
            actions = torch.empty(
                len(local_particles),
                dtype=torch.long,
            )

            for i, particle in enumerate(local_particles):
                actions[i] = particle.sample_action()

            res_q.put(
                (
                    "actions",
                    worker_id,
                    actions,
                )
            )

        # ------------------------------------------------------------------
        # AFTER
        # ------------------------------------------------------------------
        elif cmd == "after":
            o_t, a_t, r_t = msg[1:]

            results = []

            for i, particle in enumerate(local_particles):
                result = particle.after_action(
                    o_t,
                    int(a_t),
                    float(r_t),
                )

                results.append(result)
                local_logw[i] = float(particle.log_weight)

            res_q.put(
                (
                    "after_done",
                    worker_id,
                    local_logw.clone(),
                    results,
                )
            )

        # ------------------------------------------------------------------
        # GET STATES
        # ------------------------------------------------------------------
        elif cmd == "get_states":
            (req_global_i,) = msg[1:]

            global_to_local = {g: li for li, g in enumerate(particle_indices)}

            out = {}

            for g in req_global_i:
                if g in global_to_local:
                    li = global_to_local[g]
                    out[g] = local_particles[li].to_state()

            res_q.put(
                (
                    "states",
                    worker_id,
                    out,
                )
            )

        # ------------------------------------------------------------------
        # SET PARTICLES FROM STATES
        # ------------------------------------------------------------------
        elif cmd == "set_particles_from_states":
            (new_states,) = msg[1:]

            assert len(new_states) == len(local_particles)

            local_particles = [
                particle_cls.from_state(
                    hyp_params,
                    state,
                )
                for state in new_states
            ]

            # Reset particle weights after resampling.
            for particle in local_particles:
                particle.log_weight = 0.0

            local_logw.zero_()

            res_q.put(
                (
                    "set_done",
                    worker_id,
                )
            )

        # ------------------------------------------------------------------
        # GET PARTICLES
        # ------------------------------------------------------------------
        elif cmd == "get_particles":
            (req_global_i,) = msg[1:]

            global_to_local = {g: li for li, g in enumerate(particle_indices)}

            out = {}

            for g in req_global_i:
                if g in global_to_local:
                    li = global_to_local[g]
                    out[g] = local_particles[li]

            res_q.put(
                (
                    "particles",
                    worker_id,
                    out,
                )
            )

        else:
            raise ValueError(f"Unknown cmd {cmd}")


class IdealObsPFBase(SPMDWP, Subject):
    """
    Base class containing all common particle-filtering logic.

    Subclasses only need to specify:

        particle_cls
        _init_context_storage()
        _store_after_result()
    """

    particle_cls = None

    def __init__(
        self,
        N: int,
        n_workers: int,
        hyp_params: dict,
    ):
        worker_fn = partial(
            pf_worker_loop,
            particle_cls=self.particle_cls,
        )

        super().__init__(
            N,
            n_workers,
            worker_fn,
            hyp_params,
        )

        self.log_weights = torch.zeros(
            N,
            dtype=torch.float64,
        )

        self.vec_s_t = torch.zeros(
            N,
            dtype=torch.long,
        )

        self.vec_j_t = torch.zeros(
            N,
            dtype=torch.long,
        )

        self._init_context_storage(N)

    # ----------------------------------------------------------------------
    # INITIALISATION
    # ----------------------------------------------------------------------

    def _init_context_storage(self, N: int):
        """
        Initialise subclass-specific context storage.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # WEIGHTS
    # ----------------------------------------------------------------------

    @property
    def weights(self):
        """
        Weights of each particle, summing to 1.
        """
        return torch.softmax(
            self.log_weights,
            dim=0,
        )

    # ----------------------------------------------------------------------
    # BEFORE ACTION
    # ----------------------------------------------------------------------

    def before_action(self, o_t: torch.Tensor):
        """
        Process the incoming observation o_t.
        """
        self._broadcast(
            (
                "before",
                o_t,
            )
        )

        self._gather_n(
            "before_done",
            self.n_workers,
        )

    # ----------------------------------------------------------------------
    # SELECT ACTION
    # ----------------------------------------------------------------------

    def select_action(self):
        """
        Select an action a_t.
        """

        self._broadcast(("act",))

        msgs = self._gather_n(
            "actions",
            self.n_workers,
        )

        actions = torch.empty(
            self.N,
            dtype=torch.long,
        )

        for _, worker_id, acts_local in msgs:
            shard = self.shards[worker_id]
            actions[shard] = acts_local

        self._last_actions = actions.clone()

        # Select a particle according to its posterior weight.
        selected_particle = (
            D.Categorical(
                probs=self.weights,
            )
            .sample()
            .item()
        )

        selected_action = int(actions[selected_particle].item())

        return selected_action

    # ----------------------------------------------------------------------
    # AFTER ACTION
    # ----------------------------------------------------------------------

    def after_action(
        self,
        o_t: torch.Tensor,
        a_t: int,
        r_t: float,
    ):
        """
        Process the resulting observation, action, and reward.
        """

        self._broadcast(
            (
                "after",
                o_t,
                int(a_t),
                float(r_t),
            )
        )

        msgs = self._gather_n(
            "after_done",
            self.n_workers,
        )

        for _, worker_id, logw_local, results in msgs:
            shard = self.shards[worker_id]

            self.log_weights[shard] = logw_local

            for local_i, result in enumerate(results):
                global_i = shard[local_i]

                self._store_after_result(
                    global_i,
                    result,
                )

        # Resample particles with a low probability to maintain diversity.
        if bern_sample(0.02) == 1.0:
            self.resample()

    def _store_after_result(
        self,
        global_i: int,
        result,
    ):
        """
        Store the result of Particle.after_action().

        Implemented by subclasses because Particle and ParticleJCRP
        return different numbers of context variables.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # RESAMPLING
    # ----------------------------------------------------------------------

    def resample(self):
        """
        Resample particles according to their weights.
        """

        # Sample N ancestor indices using current log weights.
        ancestor = D.Categorical(
            logits=self.log_weights,
        ).sample((self.N,))

        ancestor = ancestor.to(torch.long)

        # ------------------------------------------------------------------
        # 1. Get states of all unique ancestors.
        # ------------------------------------------------------------------

        needed = ancestor.unique().tolist()

        self._broadcast(
            (
                "get_states",
                needed,
            )
        )

        state_msgs = self._gather_n(
            "states",
            self.n_workers,
        )

        ancestor_state = {}

        for _, worker_id, partial_states in state_msgs:
            ancestor_state.update(partial_states)

        missing = [i for i in needed if i not in ancestor_state]

        if missing:
            raise RuntimeError(
                f"Missing ancestor states for indices: " f"{missing[:10]} ..."
            )

        # ------------------------------------------------------------------
        # 2. Reconstruct each worker's new particle population.
        # ------------------------------------------------------------------

        for worker_id, shard in enumerate(self.shards):
            new_states = []

            for global_pos in shard:
                anc_idx = int(ancestor[global_pos].item())

                new_states.append(ancestor_state[anc_idx])

            self.cmd_qs[worker_id].put(
                (
                    "set_particles_from_states",
                    new_states,
                )
            )

        self._gather_n(
            "set_done",
            self.n_workers,
        )

        # ------------------------------------------------------------------
        # 3. Reset weights after resampling.
        # ------------------------------------------------------------------

        self.log_weights.zero_()

    # ----------------------------------------------------------------------
    # PARTICLES
    # ----------------------------------------------------------------------

    @property
    def particles(self) -> list:
        """
        Returns a snapshot list of Particle objects in global index order.

        These are deserialized copies of the particles in the worker
        processes, not live worker objects.
        """

        all_idx = list(range(self.N))

        self._broadcast(
            (
                "get_particles",
                all_idx,
            )
        )

        msgs = self._gather_n(
            "particles",
            self.n_workers,
        )

        merged = {}

        for _, worker_id, partial in msgs:
            merged.update(partial)

        missing = [i for i in all_idx if i not in merged]

        if missing:
            raise RuntimeError(f"Missing particles for indices: " f"{missing[:10]} ...")

        return [merged[i] for i in range(self.N)]

    # ----------------------------------------------------------------------
    # ACTION PROBABILITY
    # ----------------------------------------------------------------------

    @property
    def p_action(self) -> np.ndarray:
        """
        Weighted action probabilities at the current time step.
        """

        w = self.weights.to(torch.float64)

        acts = self._last_actions.to(torch.long)
        # print("w: ", w)
        # print("acts: ", acts)

        p_actions = torch.bincount(
            acts,
            weights=w,
            minlength=4,
        ).to(torch.float)

        p_actions = torch.round(
            p_actions,
            decimals=2,
        )

        return p_actions.detach().numpy()
    
    def p_action(self) -> np.ndarray:
        """
        Weighted action probabilities at the current time step, inferred from the particles' last sampled actions.
        """
        w = self.weights.to(torch.float64)  # [N], sums to 1
        
        acts = self._last_actions.to(torch.long)  # [N]
        p_actions = torch.bincount(acts, weights=w, minlength=4).to(torch.float)
        p_actions = torch.round(p_actions, decimals=2).detach().numpy()

        return p_actions

    # ----------------------------------------------------------------------
    # STATE PROBABILITY
    # ----------------------------------------------------------------------

    @property
    def p_state(self) -> np.ndarray:
        """
        Probability distribution over latent states.
        """

        w = self.weights.to(torch.float64)

        p_s = torch.bincount(
            self.vec_s_t.to(torch.long),
            weights=w,
            minlength=2,
        ).to(torch.float)

        p_s = torch.round(
            p_s,
            decimals=2,
        )

        return p_s.detach().numpy()

    # ----------------------------------------------------------------------
    # JUMP PROBABILITY
    # ----------------------------------------------------------------------

    @property
    def p_jump(self) -> float:
        """
        Probability of a jump at the current time step.
        """

        w = self.weights.to(torch.float64)

        p_jump = w[self.vec_j_t.to(torch.long) == 1].sum().to(torch.float)

        p_jump = torch.round(
            p_jump,
            decimals=2,
        )

        return float(p_jump.detach().numpy())


class IdealObsPF(IdealObsPFBase):
    """
    Particle filtering implementation of the ideal observer
    with separate observation and reward contexts.
    """

    particle_cls = Particle

    def _init_context_storage(self, N: int):
        self.vec_c_o_t = torch.zeros(
            N,
            dtype=torch.long,
        )

        self.vec_c_r_t = torch.zeros(
            N,
            dtype=torch.long,
        )

    def _store_after_result(
        self,
        global_i: int,
        result,
    ):
        s_t, c_o_t, c_r_t, j_t = result

        self.vec_s_t[global_i] = s_t
        self.vec_c_o_t[global_i] = c_o_t
        self.vec_c_r_t[global_i] = c_r_t
        self.vec_j_t[global_i] = j_t


class IdealObsPFJCRP(IdealObsPFBase):
    """
    Particle filtering implementation of the ideal observer
    with a single context.
    """

    particle_cls = ParticleJCRP

    def _init_context_storage(self, N: int):
        self.vec_c_t = torch.zeros(
            N,
            dtype=torch.long,
        )

    def _store_after_result(
        self,
        global_i: int,
        result,
    ):
        s_t, c_t, j_t = result

        self.vec_s_t[global_i] = s_t
        self.vec_c_t[global_i] = c_t
        self.vec_j_t[global_i] = j_t
