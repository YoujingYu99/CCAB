import torch
import numpy as np
import torch.distributions as D
from conjugates import (
    ConjugateModel,
    ConjugateBernoulli,
    ConjugateGaussian,
    ConjugateGaussianInvGamma,
    ConjugateOptimalArm,
)
from utils_sample import normalise, bern_sample, cat_sample, cat2D_sample
from crp import CjCRP, JCRP


class ParticleBase:
    """
    Common functionality for Particle and ParticleJCRP.
    """

    def __init__(
        self,
        hyp_niw: dict = None,
        hyp_bb: dict = None,
    ):
        """
        Initialise the common particle state.
        """
        # Store hyperparameters
        self.hyp_niw = hyp_niw
        self.hyp_bb = hyp_bb

        # Latent State Space
        self.state_space = [0, 1]
        self.state_probs = torch.tensor([0.5, 0.5])
        self.n_states = len(self.state_space)

        # Models for observations and rewards, organized by state and context
        # self.obs_models: list[list[ConjugateGaussian]] = [[] for _ in self.state_space]
        self.obs_models: list[list[ConjugateGaussianInvGamma]] = [
            [] for _ in self.state_space
        ]

        # self.rew_models: list[list[ConjugateBernoulli]] = [[] for _ in self.state_space]
        self.rew_models: list[list[ConjugateOptimalArm]] = [
            [] for _ in self.state_space
        ]

        # Novel context models (used for computing likelihood of new contexts)
        # self.novel_obs_model = ConjugateGaussian(**self.hyp_niw)
        self.novel_obs_model = ConjugateGaussianInvGamma(**self.hyp_niw)

        # self.novel_rew_model = ConjugateBernoulli(**self.hyp_bb)
        self.novel_rew_model = ConjugateOptimalArm(**self.hyp_bb)

        # Particle weight for resampling
        self.log_weight = 0.0

    def _to_state(self, crp_name):
        """
        Common state representation for reconstructing a particle.
        """
        return {
            crp_name: getattr(self, crp_name).to_state(),
            "obs_models": [
                [m.to_state() for m in self.obs_models[i]] for i in range(self.n_states)
            ],
            "rew_models": [
                [m.to_state() for m in self.rew_models[i]] for i in range(self.n_states)
            ],
            "log_weight": float(self.log_weight),
        }

    def _restore_models(self, hyp_param, state):
        """
        Restore the observation and reward models from state.
        """
        # Restore Observation Models
        self.obs_models = [[] for _ in self.state_space]
        for i in range(self.n_states):
            for m_state in state["obs_models"][i]:
                self.obs_models[i].append(
                    # ConjugateGaussian.from_state(
                    #     hyp_param["hyp_niw"],
                    #     state=m_state,
                    # )
                    ConjugateGaussianInvGamma.from_state(
                        hyp_param["hyp_niw"],
                        state=m_state,
                    )
                )

        # Restore Reward Models
        self.rew_models = [[] for _ in self.state_space]
        for i in range(self.n_states):
            for m_state in state["rew_models"][i]:
                self.rew_models[i].append(
                    # ConjugateBernoulli.from_state(
                    #     hyp_param["hyp_bb"],
                    #     state=m_state,
                    # )
                    ConjugateOptimalArm.from_state(
                        hyp_param["hyp_bb"],
                        state=m_state,
                    )
                )

        # Restore weight
        self.log_weight = float(state["log_weight"])


class Particle(ParticleBase):
    """
    Main class implementing a single particle in the CCEB model
    """

    def __init__(
        self,
        hyp_cjcrp: dict = None,
        hyp_niw: dict = None,
        hyp_bb: dict = None,
    ):
        """
        Initialise a particle with given hyperparameters and no active contexts.

        :param dict hyp_cjcrp:   hyperparameters for the coupled-jump CRP prior over contexts (gamma, alpha_o, alpha_r)
        :param dict hyp_niw:     hyperparameters for the Normal-Inverse-Wishart observation model (alpha, beta, kappa, mu)
        :param dict hyp_bb:      hyperparameters for the Beta-Bernoulli reward model (alpha, beta)
        """
        # Store hyperparameters
        self.hyp_cjcrp = hyp_cjcrp

        # Coupled jump CRP prior
        self.hyp_gamma = hyp_cjcrp["gamma"]
        self.cjcrp = CjCRP(**self.hyp_cjcrp)

        # Initialise common particle state
        super().__init__(hyp_niw, hyp_bb)

    def to_state(self):
        """
        Minimal state representation needed to reconstruct the Particle object. \\
        This consists of the state for the coupled-jump CRP prior (counts and previously active contexts), \\
        states of the observation and reward models (sufficent statistics), and the particle weight.
        """
        return self._to_state("cjcrp")

    @classmethod
    def from_state(cls, hyp_param, state):
        """
        Reconstruct the mutable state of a particle from the hyperparameters and the states of the subcomponents.\\
        Much much faster than deepcopy, and allows the particle to be transferred between processes with minimal overhead
        """
        p = cls(
            hyp_param["hyp_cjcrp"],
            hyp_param["hyp_niw"],
            hyp_param["hyp_bb"],
        )

        # restore cjCRP
        p.cjcrp = CjCRP.from_state(
            hyp_param["hyp_cjcrp"],
            state=state["cjcrp"],
        )

        p._restore_models(hyp_param, state)

        return p

    def create_new_context_models(self, h_t, c_o_t, c_r_t):
        """
        Create likelihood models for the new observation and reward contexts if needed
        """

        # New models only required if jump occurs
        if h_t == 1:

            # Create observation models for new context if needed
            if (
                c_o_t == self.cjcrp.CRP_o.n_active_contexts
            ):  # equality corresponds to new context since contexts are indexed from 0 to n-1
                for i, state in enumerate(self.state_space):
                    # self.obs_models[i].append(ConjugateGaussian(**self.hyp_niw))
                    self.obs_models[i].append(ConjugateGaussianInvGamma(**self.hyp_niw))

            # Create reward models for newcontext if needed
            if c_r_t == self.cjcrp.CRP_r.n_active_contexts:
                for i, state in enumerate(self.state_space):
                    # self.rew_models[i].append(ConjugateBernoulli(**self.hyp_bb))
                    self.rew_models[i].append(ConjugateOptimalArm(**self.hyp_bb))

    def obs_likelihood(self, o_t: torch.Tensor) -> torch.Tensor:
        """
        P(o_t | s_t=i, c^o_t=j)

            What is the likelihood of the observation o_t under each state i and observation context j?

        returns a tensor of shape (n_states, n_contexts_o + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(
            self.n_states,
            (self.cjcrp.CRP_o.n_active_contexts + 1),
        )

        for i in range(self.n_states):
            for j in range(self.cjcrp.CRP_o.n_active_contexts):
                """
                P(o_t | s_t=i, c^o_t=j) = ∫ P(o_t | ω_i,j) P(ω_i,j) dω_i,j
                """
                likelihoods[i, j] = self.obs_models[i][j].pred_lh(o_t)

            """
            P(o_t | s_t=i, c^o_t=new) = ∫ P(o_t | ω_i,new) P(ω_i,new) dω_i,new
            """
            likelihoods[i, -1] = self.novel_obs_model.pred_lh(o_t)

        return likelihoods

    def rew_likelihood(self, a_t: int, r_t: float) -> torch.Tensor:
        """
        P(r_t | s_t=i, c^r_t=k, a_t)

            Given the action a_t, what is the likelihood of the reward r_t under each state i and reward context k?

        returns a tensor of shape (n_states, n_contexts_r + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(
            self.n_states,
            (self.cjcrp.CRP_r.n_active_contexts + 1),
        )

        for i in range(self.n_states):
            for k in range(self.cjcrp.CRP_r.n_active_contexts):
                """
                P(r_t | s_t=i, c^r_t=k, a_t) = ∫ P(r_t | υ_i,k,a) P(υ_i,k,a) dυ_i,k,a
                """
                likelihoods[i, k] = self.rew_models[i][k].pred_lh(a_t, r_t)

            """
            P(r_t | s_t=i, c^r_t=new, a_t) = ∫ P(r_t | υ_i,new,a) P(υ_i,new,a) dυ_i,new,a
            """
            likelihoods[i, -1] = self.novel_rew_model.pred_lh(a_t, r_t)

        return likelihoods

    def before_action(self, o_t: torch.Tensor):
        """
        Compute
        P(S_t, C^o_t, C^r_t, H_t | o_{1:t}, a_{1:t-1}, r_{1:t-1})
        """

        # Calculate observation likelihoods for each state and observation context
        """
        P(o_t | S_t=i, C^o_t=j)
        """
        self._obs_lh = self.obs_likelihood(o_t)

        # Compute belief over states conditional on jump event and observation
        """
        P(S_t=i | o_t, H_t=0, ...) ∝ P(o_t | S_t=i, C^o_t=c_o_{t-1}) P(S_t=i)
        """
        self.state_belief_stay = normalise(
            torch.einsum(
                "i,i->i",
                self._obs_lh[:, self.cjcrp.prev_c_o],
                self.state_probs,
            )
        )

        if self.cjcrp.CRP_o.n_active_contexts == 0:
            self.state_belief_stay = torch.zeros_like(self.state_probs)

        """
        P(S_t=i | o_t, H_t=1, ...) ∝ Σ_j P(o_t | S_t=i, C^o_t=j) P(C^o_t=j | θ^o) P(S_t=i)
        """
        self.state_belief_jump = normalise(
            torch.einsum(
                "ij,j,i->i",
                self._obs_lh,
                self.cjcrp.CRP_o.probs,
                self.state_probs,
            )
        )

        """
        P(o_t | H_t=0) = Σ_i P(o_t | S_t=i, C^o_t=c_o_{t-1}) P(S_t=i)
        """
        stay_evidence = torch.einsum(
            "i,i->",
            self.state_probs,
            self._obs_lh[:, self.cjcrp.prev_c_o],
        )

        if self.cjcrp.CRP_o.n_active_contexts == 0:
            stay_evidence = torch.tensor(0.0)

        """
        P(o_t | H_t=1) = Σ_i Σ_j P(o_t | S_t=i, C^o_t=j) P(C^o_t=j | θ^o) P(S_t=i)
        """
        jump_evidence = torch.einsum(
            "ij,j,i->",
            self._obs_lh,
            self.cjcrp.CRP_o.probs,
            self.state_probs,
        )

        mixture_evidence = (
            1.0 - self.hyp_gamma
        ) * stay_evidence + self.hyp_gamma * jump_evidence

        self.log_weight += torch.log(mixture_evidence + 1e-30)

        """
        P(H_t=1 | o_t)
        """
        self.p_jump_o = torch.clamp(
            (self.hyp_gamma * jump_evidence) / (mixture_evidence + 1e-30),
            0.0,
            1.0,
        )

    def sample_action(self):
        """
        Thompson sampling using the temporary belief induced by o_t
        """
        h_hat = bern_sample(self.p_jump_o)

        if h_hat == 0:
            s_hat = cat_sample(self.state_belief_stay)
        else:
            s_hat = cat_sample(self.state_belief_jump)

        if h_hat == 0:
            c_r_hat = self.cjcrp.prev_c_r
        else:
            c_r_hat = self.cjcrp.CRP_r.sample()

        if c_r_hat == self.cjcrp.CRP_r.n_active_contexts:
            predicted_rewards = self.novel_rew_model.sample_post_dist()
        else:
            predicted_rewards = self.rew_models[s_hat][c_r_hat].sample_post_dist()

        a_t = int(torch.argmax(predicted_rewards).item())

        return a_t

    def after_action(
        self,
        o_t: torch.Tensor,
        a_t: int,
        r_t: float,
    ):
        """
        Compute
        P(S_t, C^o_t, C^r_t, H_t | o_{1:t}, a_{1:t}, r_{1:t})
        """

        self._rew_lh = self.rew_likelihood(a_t, r_t)

        stay_evidence = torch.einsum(
            "i,i->",
            self._rew_lh[:, self.cjcrp.prev_c_r],
            self.state_belief_stay,
        )

        if self.cjcrp.CRP_r.n_active_contexts == 0:
            stay_evidence = torch.tensor(0.0)

        jump_evidence = torch.einsum(
            "ik,k,i->",
            self._rew_lh,
            self.cjcrp.CRP_r.probs,
            self.state_belief_jump,
        )

        mixture_evidence = (
            1.0 - self.p_jump_o
        ) * stay_evidence + self.p_jump_o * jump_evidence

        self.log_weight += torch.log(mixture_evidence + 1e-30)

        p_jump_ora = torch.clamp(
            (self.p_jump_o * jump_evidence) / (mixture_evidence + 1e-30),
            0.0,
            1.0,
        )

        h_t = bern_sample(p_jump_ora)

        if h_t == 0:
            c_o_t, c_r_t = (
                int(self.cjcrp.prev_c_o),
                int(self.cjcrp.prev_c_r),
            )
        else:
            context_probs = torch.einsum(
                "i,ij,ik,j,k->jk",
                self.state_probs,
                self._obs_lh,
                self._rew_lh,
                self.cjcrp.CRP_o.probs,
                self.cjcrp.CRP_r.probs,
            )

            c_o_t, c_r_t = cat2D_sample(context_probs)

        state_prob = torch.einsum(
            "i,i,i->i",
            self.state_probs,
            self._rew_lh[:, c_r_t],
            self._obs_lh[:, c_o_t],
        )

        s_t = cat_sample(state_prob)

        self.create_new_context_models(h_t, c_o_t, c_r_t)

        self.obs_models[s_t][c_o_t].update(o_t)
        self.rew_models[s_t][c_r_t].update(a_t, r_t)

        self.cjcrp.update(h_t, c_o_t, c_r_t)

        return s_t, c_o_t, c_r_t, h_t


class ParticleJCRP(ParticleBase):
    """
    Main class implementing a single particle in the CAB model
    """

    def __init__(
        self,
        hyp_jcrp: dict = None,
        hyp_niw: dict = None,
        hyp_bb: dict = None,
    ):
        """
        Initialise a particle with given hyperparameters and no active contexts.

        :param dict hyp_jcrp:   hyperparameters for the jump CRP prior over contexts (gamma, alpha)
        :param dict hyp_niw:     hyperparameters for the Normal-Inverse-Wishart observation model (alpha, beta, kappa, mu)
        :param dict hyp_bb:      hyperparameters for the Beta-Bernoulli reward model (alpha, beta)
        """
        # Store hyperparameters
        self.hyp_jcrp = hyp_jcrp

        # Jump CRP prior
        self.hyp_gamma = hyp_jcrp["gamma"]
        self.jcrp = JCRP(**self.hyp_jcrp)

        # Initialise common particle state
        super().__init__(hyp_niw, hyp_bb)

    def to_state(self):
        """
        Minimal state representation needed to reconstruct the Particle object. \\
        This consists of the state for the jump CRP prior (counts and previously active contexts), \\
        states of the observation and reward models (sufficent statistics), and the particle weight.
        """
        return self._to_state("jcrp")

    @classmethod
    def from_state(cls, hyp_param, state):
        """
        Reconstruct the mutable state of a particle from the hyperparameters and the states of the subcomponents.\\
        Much much faster than deepcopy, and allows the particle to be transferred between processes with minimal overhead
        """
        p = cls(
            hyp_param["hyp_jcrp"],
            hyp_param["hyp_niw"],
            hyp_param["hyp_bb"],
        )

        # restore JCRP
        p.jcrp = JCRP.from_state(
            hyp_param["hyp_jcrp"],
            state=state["jcrp"],
        )

        p._restore_models(hyp_param, state)

        return p

    def create_new_context_models(self, h_t, c_t):
        """
        Create likelihood models for the new contexts if needed
        """

        # New models only required if jump occurs
        if h_t == 1:

            # Create observation models for new context if needed
            if (
                c_t == self.jcrp.CRP.n_active_contexts
            ):  # equality corresponds to new context since contexts are indexed from 0 to n-1
                for i, state in enumerate(self.state_space):
                    # self.obs_models[i].append(ConjugateGaussian(**self.hyp_niw))
                    self.obs_models[i].append(ConjugateGaussianInvGamma(**self.hyp_niw))

            # Create reward models for new context if needed
            if c_t == self.jcrp.CRP.n_active_contexts:
                for i, state in enumerate(self.state_space):
                    self.rew_models[i].append(ConjugateOptimalArm(**self.hyp_bb))
                    # self.rew_models[i].append(ConjugateBernoulli(**self.hyp_bb))

    def obs_likelihood(self, o_t: torch.Tensor) -> torch.Tensor:
        """
        P(o_t | s_t=i, c_t=j)

            What is the likelihood of the observation o_t under each state i and context j?

        returns a tensor of shape (n_states, n_contexts + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(
            self.n_states,
            (self.jcrp.CRP.n_active_contexts + 1),
        )

        for i in range(self.n_states):
            for j in range(self.jcrp.CRP.n_active_contexts):
                """
                P(o_t | s_t=i, c_t=j) = ∫ P(o_t | ω_i,j) P(ω_i,j) dω_i,j
                """
                likelihoods[i, j] = self.obs_models[i][j].pred_lh(o_t)

            """
            P(o_t | s_t=i, c_t=new) = ∫ P(o_t | ω_i,new) P(ω_i,new) dω_i,new
            """
            likelihoods[i, -1] = self.novel_obs_model.pred_lh(o_t)

        return likelihoods

    def rew_likelihood(self, a_t: int, r_t: float) -> torch.Tensor:
        """
        P(r_t | s_t=i, c_t=j, a_t)

            Given the action a_t, what is the likelihood of the reward r_t under each state i and context j?

        returns a tensor of shape (n_states, n_contexts + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(
            self.n_states,
            (self.jcrp.CRP.n_active_contexts + 1),
        )

        for i in range(self.n_states):
            for j in range(self.jcrp.CRP.n_active_contexts):
                """
                P(r_t | s_t=i, c_t=j, a_t) = ∫ P(r_t | υ_i,j,a) P(υ_i,j) dυ_i,j,a
                """
                likelihoods[i, j] = self.rew_models[i][j].pred_lh(a_t, int(r_t))

            """
            P(r_t | s_t=i, c_t=new, a_t) = ∫ P(r_t | υ_i,new,a) P(υ_i,new) dυ_i,new,a
            """
            likelihoods[i, -1] = self.novel_rew_model.pred_lh(a_t, int(r_t))

        return likelihoods

    def before_action(self, o_t: torch.Tensor):
        """
        Compute
        P(S_t, C_t, H_t | o_{1:t}, a_{1:t-1}, r_{1:t-1})
        """

        self._obs_lh = self.obs_likelihood(o_t)

        self.state_belief_stay = normalise(
            torch.einsum(
                "i,i->i",
                self._obs_lh[:, self.jcrp.prev_c],
                self.state_probs,
            )
        )

        if self.jcrp.CRP.n_active_contexts == 0:
            self.state_belief_stay = torch.zeros_like(self.state_probs)

        self.state_belief_jump = normalise(
            torch.einsum(
                "ij,j,i->i",
                self._obs_lh,
                self.jcrp.CRP.probs,
                self.state_probs,
            )
        )

        stay_evidence = torch.einsum(
            "i,i->",
            self.state_probs,
            self._obs_lh[:, self.jcrp.prev_c],
        )

        if self.jcrp.CRP.n_active_contexts == 0:
            stay_evidence = torch.tensor(0.0)

        jump_evidence = torch.einsum(
            "ij,j,i->",
            self._obs_lh,
            self.jcrp.CRP.probs,
            self.state_probs,
        )

        mixture_evidence = (
            1.0 - self.hyp_gamma
        ) * stay_evidence + self.hyp_gamma * jump_evidence

        self.log_weight += torch.log(mixture_evidence + 1e-30)

        self.p_jump_o = torch.clamp(
            (self.hyp_gamma * jump_evidence) / (mixture_evidence + 1e-30),
            0.0,
            1.0,
        )

    def sample_action(self):
        """
        Thompson sampling using the temporary belief induced by o_t
        """
        h_hat = bern_sample(self.p_jump_o)

        if h_hat == 0:
            s_hat = cat_sample(self.state_belief_stay)
        else:
            s_hat = cat_sample(self.state_belief_jump)

        if h_hat == 0:
            c_hat = self.jcrp.prev_c
        else:
            c_hat = self.jcrp.CRP.sample()

        if c_hat == self.jcrp.CRP.n_active_contexts:
            predicted_rewards = self.novel_rew_model.sample_post_dist()
        else:
            predicted_rewards = self.rew_models[s_hat][c_hat].sample_post_dist()

        a_t = int(torch.argmax(predicted_rewards).item())

        return a_t

    def after_action(
        self,
        o_t: torch.Tensor,
        a_t: int,
        r_t: float,
    ):
        """
        Compute
        P(S_t, C_t, H_t | o_{1:t}, a_{1:t}, r_{1:t})
        """

        self._rew_lh = self.rew_likelihood(a_t, r_t)

        # 1) Sample Jump
        stay_evidence = torch.einsum(
            "i,i->",
            self._rew_lh[:, self.jcrp.prev_c],
            self.state_belief_stay,
        )

        if self.jcrp.CRP.n_active_contexts == 0:
            stay_evidence = torch.tensor(0.0)

        jump_evidence = torch.einsum(
            "ij,j,i->",
            self._rew_lh,
            self.jcrp.CRP.probs,
            self.state_belief_jump,
        )

        mixture_evidence = (
            1.0 - self.p_jump_o
        ) * stay_evidence + self.p_jump_o * jump_evidence

        self.log_weight += torch.log(mixture_evidence + 1e-30)

        p_jump_ora = torch.clamp(
            (self.p_jump_o * jump_evidence) / (mixture_evidence + 1e-30),
            0.0,
            1.0,
        )

        h_t = bern_sample(p_jump_ora)

        # 2) Sample contexts given jump
        if h_t == 0:
            c_t = int(self.jcrp.prev_c)
        else:
            context_prob = torch.einsum(
                "i,ij,ij,j->j",
                self.state_probs,
                self._obs_lh,
                self._rew_lh,
                self.jcrp.CRP.probs,
            )

            c_t = cat_sample(context_prob)

        # 3) sample state given committed contexts:
        state_prob = torch.einsum(
            "i,i,i->i",
            self.state_probs,
            self._rew_lh[:, c_t],
            self._obs_lh[:, c_t],
        )

        s_t = cat_sample(state_prob)

        # ---------------------
        #   Update parameters
        # ---------------------
        self.create_new_context_models(h_t, c_t)

        self.obs_models[s_t][c_t].update(o_t)
        self.rew_models[s_t][c_t].update(a_t, r_t)

        self.jcrp.update(h_t, c_t)

        return s_t, c_t, h_t


def print_particle_params(particle):
    """
    Print the parameters of the particle's models in a readable format.

    Works for both Particle and ParticleJCRP.
    """

    print("\n> Log weight:", particle.log_weight)

    # Determine whether the particle uses separate observation/reward CRPs
    # (Particle) or a shared CRP (ParticleJCRP)
    if isinstance(particle, Particle):
        obs_crp = particle.cjcrp.CRP_o
        rew_crp = particle.cjcrp.CRP_r
    elif isinstance(particle, ParticleJCRP):
        obs_crp = particle.jcrp.CRP
        rew_crp = particle.jcrp.CRP

    # Extract and print the CRP probabilities and sufficient statistics for observation contexts
    print("\n> Observation Context CRP:")
    o_prob = obs_crp.probs.detach().numpy()
    o_prob = np.array2string(o_prob, formatter={"float_kind": lambda x: f"{x: .2f}"})
    o_ss = obs_crp.counts.detach().numpy()[: obs_crp.n_active_contexts]
    print("\nObser. Context CRP Probabilities:", o_prob, "Suff Stats:", o_ss)

    # Extract and print observation models
    print("\n> Observation Models:")
    for j in range(obs_crp.n_active_contexts):
        for i, state in enumerate(particle.state_space):
            # Extract and print format the observation model parameters (df, loc)
            observation_model_params = particle.obs_models[i][j]._pred_dist_params()
            df = observation_model_params["df"]
            loc = observation_model_params["loc"].detach().numpy()
            loc = np.array2string(loc, formatter={"float_kind": lambda x: f"{x: .2f}"})

            print(
                f"State {state}, Observ Context {j}, "
                f"Observation Model Params: df: {df}, loc: {loc}"
            )

    # Extract and print the CRP probabilities and sufficient statistics for reward contexts
    print("\n> Reward Context CRP:")
    r_prob = rew_crp.probs.detach().numpy()
    r_prob = np.array2string(r_prob, formatter={"float_kind": lambda x: f"{x: .2f}"})
    r_ss = rew_crp.counts.detach().numpy()[: rew_crp.n_active_contexts]
    print("\nReward Context CRP Probabilities:", r_prob, "Suff Stats:", r_ss)

    # Extract and print reward models
    print("\n> Reward Models:")
    for j in range(rew_crp.n_active_contexts):
        for i, state in enumerate(particle.state_space):
            # Extract and print format the reward model mean and sufficient statistics
            reward_model_mean = (
                particle.rew_models[i][j]._pred_dist_params().detach().numpy()
            )
            reward_model_mean = np.array2string(
                reward_model_mean,
                formatter={"float_kind": lambda x: f"{x: .2f}"},
            )

            suff_stats_a = (
                particle.rew_models[i][j].post_params()["alpha_n"].detach().numpy()
            )
            suff_stats_b = (
                particle.rew_models[i][j].post_params()["beta_n"].detach().numpy()
            )

            suff_stats_a = np.round(suff_stats_a, 2)
            suff_stats_b = np.round(suff_stats_b, 2)

            print(
                f"State {state}, Reward Context {j}, "
                f"Reward Model Mean: {reward_model_mean}, "
                f"Suff Stats: {suff_stats_a}, {suff_stats_b}"
            )
