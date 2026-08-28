from copy import deepcopy

import torch
import torch.distributions as D
import numpy as np

from utils import niceprint
from subject import Subject
from idealobserver import IdealObsPFJCRP, IdealObsPF
from utils_sample import cat_sample


class ExperimentHistory:
    """
    Class to store the history of an experiment, tracking variables available to the experimenter
    """

    def __init__(self):
        # Initialise the variables to be tracked
        self.state: list[int] = []  # true latent state (not observable to participant)
        self.context_o: list[list[int]] = (
            []
        )  # true context for observation (not observable to participant)
        self.context_r: list[list[int]] = (
            []
        )  # true context for reward (not observable to participant)
        self.observation: list[torch.Tensor] = []  # observation passed to participant
        self.optimal_action: list[int] = (
            []
        )  # optimal action given the true state and reward context (not observable to participant)
        self.action: list[int] = []  # action taken by participant
        self.reward: list[float] = []  # reward given to participant

    def append(
        self, state, context_o, context_r, observation, optimal_action, action, reward
    ):
        """
        Append a new timestep to the history, with the given variables
        """
        self.state.append(state)
        self.context_o.append(context_o)
        self.context_r.append(context_r)
        self.observation.append(observation)
        self.optimal_action.append(optimal_action)
        self.action.append(action)
        self.reward.append(reward)

    def __len__(self):
        """
        Returns the number of timesteps in the history
        """
        return len(self.state)


class SubjectHistory:
    """
    Class to store the history of a participant's internal variables, which are not directly observable to the experimenter
    """

    def __init__(self):
        self.p_action: list[np.ndarray] = []  # probability of each action being chosen

    def append(self, p_action):
        """
        Append a new timestep to the history, with the given variables
        """
        self.p_action.append(p_action)

    def __len__(self):
        return len(self.p_action)


class IOPFSubjectHistory(SubjectHistory):
    """
    Class to store the history of a participant's internal variables, which are not directly observable to the experimenter
    """

    def __init__(self):
        super().__init__()  # probability of each action being chosen is tracked in the base class
        self.p_state: list[np.ndarray] = []  # probability of each latent state
        self.p_jump: list[np.ndarray] = []  # probability of a jump
        self.context_o: list[list[int]] = []  # sampled context_o for each particle
        self.context_r: list[list[int]] = []  # sampled context_r for each particle
        self.weights: list[np.ndarray] = []  # weights of each particle

    def append(self, p_action, p_state, p_jump, context_o, context_r, weights):
        """
        Append a new timestep to the history, with the given variables
        """
        super().append(p_action)  # action probabilities are tracked in the base class
        self.p_state.append(p_state)
        self.p_action.append(p_action)
        self.p_jump.append(p_jump)
        self.context_o.append(context_o)
        self.context_r.append(context_r)
        self.weights.append(weights)


class ExperimentalEnvBase:
    """
    Base class for experimental environments.

    Contains functionality shared across environments:
    - trial input validation
    - reward generation
    - trial execution
    - subject/experiment history recording
    - printing
    """

    def _check_trial_inputs(
        self,
        subject: Subject,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level: int,
    ):
        assert isinstance(
            subject, Subject
        ), "Subject must be an instance of the Subject class or its subclasses"

        assert isinstance(
            experiment_history, ExperimentHistory
        ), "experiment_history must be an instance of the ExperimentHistory class"

        assert isinstance(
            subject_history, SubjectHistory
        ), "subject_history must be an instance of the SubjectHistory class or its subclasses"

        assert print_level in [
            0,
            1,
            2,
            3,
        ], "print_level must be an integer between 0 and 3, inclusive"

    def after_action(self, c, s_t, a_t):
        """
        Generate reward and optimal action given the context, state and action.
        """
        rew_params = self.context_r_params[c][s_t]

        r_t = D.Bernoulli(probs=rew_params["p_rew"][a_t]).sample().item()

        opt_a_t = torch.argmax(rew_params["p_rew"]).item()

        return r_t, opt_a_t

    def _run_trial(
        self,
        subject: Subject,
        c_o: int,
        c_r: int,
        s_t,
        o_t,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level: int,
    ):
        """
        Run the common part of a trial after the environment has generated
        the state and observation.
        """

        # -----------------
        #   Before action
        # -----------------
        if print_level > 1:
            print(
                f"\tTrue state: {s_t}, Obs.: {niceprint(o_t, 2)}",
                end="",
            )

        # Subject processes observation
        subject.before_action(o_t)

        # ------------------------
        #   Action selection
        # ------------------------
        a_t = subject.select_action()

        # ----------------
        #   After action
        # ----------------
        r_t, opt_a_t = self.after_action(c_r, s_t, a_t)

        if print_level > 1:
            print(f", Action: {a_t}, Reward: {r_t}", end="")

        # Subject updates internal representations
        subject.after_action(o_t, a_t, r_t)

        # ----------------
        #   Record keeping
        # ----------------
        experiment_history.append(
            s_t,
            c_o,
            c_r,
            o_t,
            opt_a_t,
            a_t,
            r_t,
        )

        # Record keeping from the subject's perspective
        if isinstance(subject, IdealObsPF) and isinstance(
            subject_history, IOPFSubjectHistory
        ):
            subject_history.append(
                p_action=subject.p_action,
                p_state=subject.p_state,
                p_jump=subject.p_jump,
                context_o=deepcopy(subject.vec_c_o_t),
                context_r=deepcopy(subject.vec_c_r_t),
                weights=deepcopy(subject.weights),
            )

        elif isinstance(subject, IdealObsPFJCRP) and isinstance(
            subject_history, IOPFSubjectHistory
        ):
            subject_history.append(
                p_action=subject.p_action,
                p_state=subject.p_state,
                p_jump=subject.p_jump,
                context_o=deepcopy(subject.vec_c_t),
                context_r=deepcopy(subject.vec_c_t),
                weights=deepcopy(subject.weights),
            )

        else:
            subject_history.append(p_action=subject.p_action)

        # ----------------
        #   Printing
        # ----------------
        if print_level > 2:
            if isinstance(subject, IdealObsPFJCRP):
                print(
                    f", Action probs.: {niceprint(subject.p_action, 2)}",
                    end="",
                )

        if print_level > 1:
            print("\n", end="")


class ExperimentalEnv(ExperimentalEnvBase):
    """
    Class to represent the experimental environment.

    Observation and reward contexts are defined independently.
    """

    def __init__(
        self,
        context_o_params: dict = None,
        context_r_params: dict = None,
    ):
        """
        Observation and reward contexts are defined by their parameters,
        which are provided at initialisation.

        :param dict[int, dict] context_o_params:
            Parameters for observation contexts, of the form
            {context_id: {state_id: {param_name: param_value}}}

        :param dict[int, dict] context_r_params:
            Parameters for reward contexts, of the form
            {context_id: {state_id: {param_name: param_value}}}
        """

        self.state_probs = torch.tensor([0.5, 0.5])

        # -------------------------
        #   Observation contexts
        # -------------------------
        assert type(context_o_params) == dict, (
            "context_o_params must be a dictionary of the form "
            "{context_id: context_parameters}"
        )

        assert all(
            isinstance(k, int) for k in context_o_params.keys()
        ), "context_o_params keys must be integers representing context IDs"

        assert all(isinstance(v, dict) for v in context_o_params.values()), (
            "context_o_params values must be dictionaries representing "
            "context parameters"
        )

        self.contexts_o = list(context_o_params.keys())

        for context_id, context_params in context_o_params.items():

            assert all(isinstance(k, int) for k in context_params.keys()), (
                f"State IDs in context_o_params for context {context_id} "
                "must be integers"
            )

            assert all(isinstance(v, dict) for v in context_params.values()), (
                f"Parameters for each state in context_o_params for "
                f"context {context_id} must be dictionaries"
            )

            for state_id, state_params in context_params.items():

                assert "loc" in state_params and "cov" in state_params, (
                    f"Parameters for state {state_id} in context_o_params "
                    f"for context {context_id} must include 'loc' and 'cov'"
                )

                assert isinstance(
                    state_params["loc"],
                    torch.Tensor,
                ), (
                    f"'loc' parameter for state {state_id} in "
                    f"context_o_params for context {context_id} must be "
                    "a torch.Tensor"
                )

                assert isinstance(
                    state_params["cov"],
                    torch.Tensor,
                ), (
                    f"'cov' parameter for state {state_id} in "
                    f"context_o_params for context {context_id} must be "
                    "a torch.Tensor"
                )

        self.context_o_params: dict[int, dict] = context_o_params

        # -------------------------
        #   Reward contexts
        # -------------------------
        assert type(context_r_params) == dict, (
            "context_r_params must be a dictionary of the form "
            "{context_id: context_parameters}"
        )

        assert all(
            isinstance(k, int) for k in context_r_params.keys()
        ), "context_r_params keys must be integers representing context IDs"

        assert all(isinstance(v, dict) for v in context_r_params.values()), (
            "context_r_params values must be dictionaries representing "
            "context parameters"
        )

        self.contexts_r = list(context_r_params.keys())

        for context_id, context_params in context_r_params.items():

            assert all(isinstance(k, int) for k in context_params.keys()), (
                f"State IDs in context_r_params for context {context_id} "
                "must be integers"
            )

            assert all(isinstance(v, dict) for v in context_params.values()), (
                f"Parameters for each state in context_r_params for "
                f"context {context_id} must be dictionaries"
            )

            for state_id, state_params in context_params.items():

                assert "p_rew" in state_params, (
                    f"Parameters for state {state_id} in context_r_params "
                    f"for context {context_id} must include 'p_rew'"
                )

                assert all(
                    isinstance(v, torch.Tensor) for v in state_params.values()
                ), (
                    f"Reward probabilities in context_r_params for state "
                    f"{state_id} in context {context_id} must be torch.Tensors"
                )

        self.context_r_params: dict[int, dict] = context_r_params

    def before_action(self, c):
        """
        Sample state and observation given the observation context.
        """

        # Sample state i.i.d. for each trial
        s_t = cat_sample(self.state_probs)

        # Sample observation given the state and observation context
        obs_params = self.context_o_params[c][s_t]

        o_t = D.MultivariateNormal(
            loc=obs_params["loc"],
            covariance_matrix=obs_params["cov"],
        ).sample()

        return s_t, o_t

    def trial_step(
        self,
        subject: Subject,
        c_o: int,
        c_r: int,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level: int = 0,
    ):
        # -------------------------
        #   Checks and assertions
        # -------------------------
        self._check_trial_inputs(
            subject,
            experiment_history,
            subject_history,
            print_level,
        )

        assert c_o in self.contexts_o, (
            "Observation context must be one of the contexts defined "
            "in the environment"
        )

        assert c_r in self.contexts_r, (
            "Reward context must be one of the contexts defined " "in the environment"
        )

        # -----------------
        #   Before action
        # -----------------
        s_t, o_t = self.before_action(c_o)

        # Run common trial logic
        self._run_trial(
            subject,
            c_o,
            c_r,
            s_t,
            o_t,
            experiment_history,
            subject_history,
            print_level,
        )

    def repeat_trials(
        self,
        subject: Subject,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        N: int,
        c_o: int,
        c_r: int,
        print_level: int = 0,
    ):
        """
        Run a batch of trials with the given context combination.
        """

        if print_level > 0:
            print(
                f"\n << Running {N} trials with true obs. context = "
                f"{c_o} and true rew. context = {c_r} >> \n"
            )

        for i in range(N):

            if print_level > 1:
                print(
                    f"Step {i + 1}:   ",
                    end="",
                )

            self.trial_step(
                subject,
                c_o,
                c_r,
                experiment_history,
                subject_history,
                print_level,
            )

        if print_level > 0:
            print("\n >> Batch completed \n")


class ExperimentalEnvCRP(ExperimentalEnvBase):
    """
    Class to represent the experimental environment.

    Observation and reward contexts are coupled and therefore share
    the same context ID.
    """

    def __init__(
        self,
        context_o_params: dict = None,
        context_r_params: dict = None,
    ):
        """
        Observation and reward contexts are defined by their parameters,
        which are provided at initialisation.

        :param dict[int, dict] context_o_params:
            Parameters for observation contexts, of the form
            {context_id: {state_id: {param_name: param_value}}}

        :param dict[int, dict] context_r_params:
            Parameters for reward contexts, of the form
            {context_id: {state_id: {param_name: param_value}}}
        """

        self.state_probs = torch.tensor([0.5, 0.5])

        # -------------------------
        #   Observation contexts
        # -------------------------
        assert type(context_o_params) == dict, (
            "context_o_params must be a dictionary of the form "
            "{context_id: context_parameters}"
        )

        assert all(
            isinstance(k, int) for k in context_o_params.keys()
        ), "context_o_params keys must be integers representing context IDs"

        assert all(isinstance(v, dict) for v in context_o_params.values()), (
            "context_o_params values must be dictionaries representing "
            "context parameters"
        )

        self.contexts_o = list(context_o_params.keys())

        for context_id, context_params in context_o_params.items():

            assert all(isinstance(k, int) for k in context_params.keys()), (
                f"State IDs in context_o_params for context {context_id} "
                "must be integers"
            )

            assert all(isinstance(v, dict) for v in context_params.values()), (
                f"Parameters for each state in context_o_params for "
                f"context {context_id} must be dictionaries"
            )

            for state_id, state_params in context_params.items():

                assert "loc" in state_params and "cov" in state_params, (
                    f"Parameters for state {state_id} in context_o_params "
                    f"for context {context_id} must include 'loc' and 'cov'"
                )

                assert isinstance(
                    state_params["loc"],
                    torch.Tensor,
                ), (
                    f"'loc' parameter for state {state_id} in "
                    f"context_o_params for context {context_id} must be "
                    "a torch.Tensor"
                )

                assert isinstance(
                    state_params["cov"],
                    torch.Tensor,
                ), (
                    f"'cov' parameter for state {state_id} in "
                    f"context_o_params for context {context_id} must be "
                    "a torch.Tensor"
                )

        self.context_o_params: dict[int, dict] = context_o_params

        # -------------------------
        #   Reward contexts
        # -------------------------
        assert type(context_r_params) == dict, (
            "context_r_params must be a dictionary of the form "
            "{context_id: context_parameters}"
        )

        assert all(
            isinstance(k, int) for k in context_r_params.keys()
        ), "context_r_params keys must be integers representing context IDs"

        assert all(isinstance(v, dict) for v in context_r_params.values()), (
            "context_r_params values must be dictionaries representing "
            "context parameters"
        )

        self.contexts_r = list(context_r_params.keys())

        for context_id, context_params in context_r_params.items():

            assert all(isinstance(k, int) for k in context_params.keys()), (
                f"State IDs in context_r_params for context {context_id} "
                "must be integers"
            )

            assert all(isinstance(v, dict) for v in context_params.values()), (
                f"Parameters for each state in context_r_params for "
                f"context {context_id} must be dictionaries"
            )

            for state_id, state_params in context_params.items():

                assert "p_rew" in state_params, (
                    f"Parameters for state {state_id} in context_r_params "
                    f"for context {context_id} must include 'p_rew'"
                )

                assert all(
                    isinstance(v, torch.Tensor) for v in state_params.values()
                ), (
                    f"Reward probabilities in context_r_params for state "
                    f"{state_id} in context {context_id} must be torch.Tensors"
                )

        self.context_r_params: dict[int, dict] = context_r_params

    def before_action(self, c):
        """
        Sample state and observation given the context.
        """

        # Sample state i.i.d. for each trial
        s_t = cat_sample(self.state_probs)

        # Sample observation given the state and observation context
        obs_params = self.context_o_params[c][s_t]

        o_t = D.MultivariateNormal(
            loc=obs_params["loc"],
            covariance_matrix=obs_params["cov"],
        ).sample()

        return s_t, o_t

    def trial_step(
        self,
        subject: Subject,
        c: int,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level: int = 0,
    ):
        # -------------------------
        #   Checks and assertions
        # -------------------------
        self._check_trial_inputs(
            subject,
            experiment_history,
            subject_history,
            print_level,
        )

        assert c in self.contexts_o, (
            "Observation context must be one of the contexts defined "
            "in the environment"
        )

        assert c in self.contexts_r, (
            "Reward context must be one of the contexts defined " "in the environment"
        )

        # -----------------
        #   Before action
        # -----------------
        s_t, o_t = self.before_action(c)

        # Run common trial logic
        self._run_trial(
            subject,
            c,
            c,
            s_t,
            o_t,
            experiment_history,
            subject_history,
            print_level,
        )

    def repeat_trials(
        self,
        subject: Subject,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        N: int,
        c: int,
        print_level: int = 0,
    ):
        """
        Run a batch of trials with the given context.
        """

        if print_level > 0:
            print(f"\n << Running {N} trials with true context = {c} >> \n")

        for i in range(N):

            if print_level > 1:
                print(
                    f"Step {i + 1}:   ",
                    end="",
                )

            self.trial_step(
                subject,
                c,
                experiment_history,
                subject_history,
                print_level,
            )

        if print_level > 0:
            print("\n >> Batch completed \n")


class ExperimentalCRP(ExperimentalEnvBase):
    """
    Class to represent the experimental environment.

    Observations, states and contexts are given rather than sampled.
    """

    def __init__(
        self,
        context_r_params: dict = None,
        contexts: np.ndarray = None,
        states: np.ndarray = None,
        observations: np.ndarray = None,
    ):
        """
        :param dict[int, dict] context_r_params:
            Parameters for reward contexts, of the form
            {context_id: {state_id: {param_name: param_value}}}

        :param np.ndarray contexts:
            Integer-valued contexts.

        :param np.ndarray states:
            Integer-valued states.

        :param np.ndarray observations:
            Float-valued observations.
        """

        # -------------------------
        #   Reward contexts
        # -------------------------
        assert type(context_r_params) == dict, (
            "context_r_params must be a dictionary of the form "
            "{context_id: context_parameters}"
        )

        assert all(
            isinstance(k, int) for k in context_r_params.keys()
        ), "context_r_params keys must be integers representing context IDs"

        assert all(isinstance(v, dict) for v in context_r_params.values()), (
            "context_r_params values must be dictionaries representing "
            "context parameters"
        )

        self.contexts_r = list(context_r_params.keys())

        for context_id, context_params in context_r_params.items():

            assert all(isinstance(k, int) for k in context_params.keys()), (
                f"State IDs in context_r_params for context {context_id} "
                "must be integers"
            )

            assert all(isinstance(v, dict) for v in context_params.values()), (
                f"Parameters for each state in context_r_params for "
                f"context {context_id} must be dictionaries"
            )

            for state_id, state_params in context_params.items():

                assert "p_rew" in state_params, (
                    f"Parameters for state {state_id} in context_r_params "
                    f"for context {context_id} must include 'p_rew'"
                )

                assert all(
                    isinstance(v, torch.Tensor) for v in state_params.values()
                ), (
                    f"Reward probabilities in context_r_params for state "
                    f"{state_id} in context {context_id} must be torch.Tensors"
                )

        self.context_r_params: dict[int, dict] = context_r_params

        # -------------------------
        #   Pre-generated data
        # -------------------------
        self.contexts = contexts.astype(int)
        self.states = states.astype(int)
        self.observations = torch.from_numpy(observations)

        self.T = contexts.shape[0]

    def before_action(self, t):
        """
        Return the pre-generated context, state and observation.
        """

        c_t = self.contexts[t]
        s_t = self.states[t]
        o_t = self.observations[t]

        return c_t, s_t, o_t

    def trial_step(
        self,
        subject: Subject,
        t: int,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level: int = 0,
    ):
        # -------------------------
        #   Checks and assertions
        # -------------------------
        self._check_trial_inputs(
            subject,
            experiment_history,
            subject_history,
            print_level,
        )

        # -----------------
        #   Before action
        # -----------------
        c_t, s_t, o_t = self.before_action(t)

        # Run common trial logic
        self._run_trial(
            subject,
            c_t,
            c_t,
            s_t,
            o_t,
            experiment_history,
            subject_history,
            print_level,
        )

    def episode(
        self,
        subject: Subject,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level: int = 0,
    ):
        """
        Run an episode of trials.
        """

        for t in range(self.T):

            if print_level > 1:
                print(
                    f"Step {t}:   ",
                    end="",
                )

            self.trial_step(
                subject,
                t,
                experiment_history,
                subject_history,
                print_level,
            )

        if print_level > 0:
            print("\n >> Episode completed \n")


class CombinedHistory:
    """
    Class to perform analysis on the experimental and subject history
    """

    def __init__(
        self, experiment_history: ExperimentHistory, subject_history: SubjectHistory
    ):
        self.experiment_history = deepcopy(experiment_history)
        self.subject_history = deepcopy(subject_history)

    @property
    def p_optimal_action(self):
        """
        Calculate the probability of optimal actions being taken by the subject at each time step
        """
        optimal_actions = np.array(self.experiment_history.optimal_action)
        subject_actions_probs = np.array(self.subject_history.p_action)

        output = np.zeros(len(optimal_actions))
        for t in range(len(optimal_actions)):
            output[t] = subject_actions_probs[t][optimal_actions[t]]

        return output
