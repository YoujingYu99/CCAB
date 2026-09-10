from typing import TypedDict

import torch
import numpy as np
import torch.distributions as D

from models_obs_rew import ConjugateModel
from models_context import ContextModel
from subject import IdealObsParams
from utils_sample import normalise, bern_sample, cat_sample, cat2D_sample


"""
Particle = IdealObsParams + ParticleState
"""
class ParticleState(TypedDict):
    """
    TypedDict for the state of a Particle object.
    """
    context_model_state: dict
    obs_model_states: list[list[dict]]
    rew_model_states: list[list[dict]]
    log_weight: float



class Particle():
    """
    Main class implementing a single particle in the various bandit models.
    """
    def __init__(self, particle_params: IdealObsParams):
        """
        Initialise a particle
        """
        # Set types of the various distributions
        self.type_context = particle_params["type_context"]
        CtxModel          = particle_params["type_context"]
        self.type_obs     = particle_params["type_obs"]
        ObsModel          = particle_params["type_obs"]
        self.type_rew     = particle_params["type_rew"]
        RewModel          = particle_params["type_rew"]

        # Store hyperparameters
        self.hyp_context = particle_params["hyp_context"]
        self.hyp_obs     = particle_params["hyp_obs"]
        self.hyp_rew     = particle_params["hyp_rew"]

        # Context Model
        self.context_model = CtxModel(**self.hyp_context)

        # Latent State Space
        self.state_space = [0, 1]
        self.state_probs = torch.tensor([0.5, 0.5])
        self.n_states = len(self.state_space)

        # Models for observations and rewards, organized by state and context (this starts empty and is populated as new contexts are discovered)
        self.obs_models : list[list[ConjugateModel]] = [[] for _ in self.state_space]
        self.rew_models : list[list[ConjugateModel]] = [[] for _ in self.state_space]

        # Novel context models (used for computing likelihood of new contexts)
        self.novel_obs_model = ObsModel(**self.hyp_obs)
        self.novel_rew_model = RewModel(**self.hyp_rew)

        # Particle weight for resampling
        self.log_weight = 0.0


    def to_state(self):
        """
        Minimal state representation needed to reconstruct the Particle object. \\
        This consists of the state for the context model, \\
        states of the observation and reward models (sufficent statistics), and the particle weight. 
        """
        return {
            "context_model_state":  self.context_model.to_state(),

            "obs_model_states":     [[m.to_state() for m in self.obs_models[i]] for i in range(self.n_states)],
            "rew_model_states":     [[m.to_state() for m in self.rew_models[i]] for i in range(self.n_states)],

            "log_weight":     float(self.log_weight),
        }

    @classmethod
    def from_state(cls, particle_params: IdealObsParams, state : ParticleState):
        """
        Reconstruct the mutable state of a particle from the type-hyperparameters and the states of the subcomponents.\\
        Much much faster than deepcopy, and allows the particle to be transferred between processes with minimal overhead
        """
        # 0) Start by reconstructing the particle with the type and hyperparameters
        p = cls(particle_params)        


        # Read out key types, hyperparameters, and states for simpler syntax below
        CtxModel     = p.type_context
        ObsModel     = p.type_obs
        RewModel     = p.type_rew

        hyp_context : dict = p.hyp_context
        hyp_obs     : dict = p.hyp_obs
        hyp_rew     : dict = p.hyp_rew

        context_state : dict             = state["context_model_state"]
        obs_states    : list[list[dict]] = state["obs_model_states"]
        rew_states    : list[list[dict]] = state["rew_model_states"]

        # 1) Restore context model
        p.context_model = CtxModel.from_state(hyp_context, state=context_state)

        # 2) Restore Observation Models
        p.obs_models = [[] for _ in p.state_space]
        for i in range(p.n_states):
            for m_state in obs_states[i]:
                p.obs_models[i].append(ObsModel.from_state(hyp_obs, state=m_state))

        # 3) Restore Reward Models
        p.rew_models = [[] for _ in p.state_space]
        for i in range(p.n_states):
            for m_state in rew_states[i]:
                p.rew_models[i].append(RewModel.from_state(hyp_rew, state=m_state))

        # 4) Restore weight
        p.log_weight = float(state["log_weight"])

        return p


    def before_action(self, o_t: torch.Tensor):
        """
        Compute the temporary belief P̄ given the newest observation o_t, and the previous sufficient statistics of the particle.

        P̄^(i)( S_t=s, C_t=c | o_t, {ξ, υ, ω}^(i)_{t-1} )
        """

        # Calculate observation likelihoods for each state and context hypothesis
        self._obs_lh = self.obs_likelihood(o_t) # [n_states, n_hypotheses]

        # Compute the unnormalised belief
        """
        P̄^(i)( s, c | o_t, {ξ, υ, ω}^(i)_{t-1} )
            ∝ P_S( s ) L_O^(i)( o_t | s, c, ω^(i)_{t-1} ) P_C^(i)( C_t=c | ξ^(i)_{t-1} )
        """
        unnorm_belief = torch.einsum('s,sc,c->sc', self.state_probs, self._obs_lh, self.context_model.hypothesis_probs)# [n_states, n_contexts_o + 1]

        # Calculate the evidence for the observation
        Z_O = torch.sum(unnorm_belief)

        # Normalize the belief to get the posterior probabilities
        self.temp_belief = unnorm_belief / (Z_O + 1e-30)

        # Adjust the particle's log weight based on the evidence 
        self.log_weight += torch.log(Z_O + 1e-30)

    def sample_action(self):
        """
        Thompson sampling using the temporary belief induced by o_t
        """

        # Sample state and context
        """
        s_hat, c_hat ~ P̄^(i)( S_t=s, C_t=c | o_t, {ξ, υ, ω}^(i)_{t-1} )
        """
        s_hat, c_hat = cat2D_sample(self.temp_belief)
        # select the reward model associated with the sampled context
        c_r_hat = self.context_model.hypothesis_r(c_hat) 

        # Sample reward parameters for (s_hat, c_r_hat)
        """
        υ_hat ~ P(ϒ | s_hat, c_r_hat)
        """
        if c_r_hat == self.n_rew_models: # equality corresponds to new model since models are indexed from 0 to n-1
            predicted_rewards = self.novel_rew_model.sample_post_dist() # special case for novel model
        else:
            predicted_rewards = self.rew_models[s_hat][c_r_hat].sample_post_dist()
        
        # Select best action
        """
        a = argmax_a 𝔼[r_t | a, s_hat, c_r_hat, υ_hat]
        """
        a_t = int(torch.argmax(predicted_rewards).item())

        return a_t

    def after_action(self, o_t: torch.Tensor, a_t: int, r_t: float):
        """
        Compute
        P( suffstat | h_t)
        """
        # ----------------------------------------------
        #   Calculate posterior over latent variables
        # ----------------------------------------------
        # Calculate reward likelihoods for each state and context hypothesis
        self._rew_lh = self.rew_likelihood(a_t, r_t)  # [n_states, n_hypotheses]

        # compute the unnormalised posterior
        """
        P(s, c | {o, a, r}_t, {ξ, υ, ω}^(i)_{t-1} )
            ∝ L_R^(i)( r_t | s, c, a_t, υ^(i)_{t-1} ) P̄^(i)( s, c | o_t, {ξ, υ, ω}^(i)_{t-1} ) 
        """
        unnorm_post = torch.einsum('sc,sc->sc',  self._rew_lh, self.temp_belief) # [n_states, n_hypotheses]

        # Compute the evidence for the reward
        Z_R = torch.sum(unnorm_post)

        # adjust the particle's log weight based on the evidence for the reward
        self.log_weight += torch.log(Z_R + 1e-30)

        # Sample state and contexts from the posterior
        """
        s_t, c_o_t ~ P(s, c | {o, a, r}_t, {ξ, υ, ω}^(i)_{t-1} )
        """
        posterior = unnorm_post / (Z_R + 1e-30)
        s_t, c_t = cat2D_sample(posterior)


        # ---------------------
        #   Update parameters 
        # --------------------- 
        j_t   = self.context_model.hypothesis_jump(c_t) # jump indicator for the sampled context
        c_o_t = self.context_model.hypothesis_o(c_t)
        c_r_t = self.context_model.hypothesis_r(c_t)

        # 1) Create new likelihood models if neeeded
        self.create_new_context_models(c_o_t, c_r_t)
        
        # 2) Update observation model sufficient statistics for the sampled context and state
        self.obs_models[s_t][c_o_t].update(o_t)
        
        # 3) Update reward model sufficient statistics for the sampled context and state
        self.rew_models[s_t][c_r_t].update(a_t, r_t)
        
        # 4) Update context model sufficient statistics given the sampled context
        self.context_model.update(c_t)

        return s_t, c_o_t, c_r_t, j_t



    def obs_likelihood(self, o_t : torch.Tensor) -> torch.Tensor:
        """
        L_O^(i)( o_t | S=s, C=c, ω^(i)_{t-1} )

            What is the likelihood of the observation o_t under each state l and observation context m?

        returns a tensor of shape (n_states, n_hypotheses) containing the likelihoods for each state and context (including a new context)
        """
        # first calculate under each observation model for each state
        """
        L_O^(i)( o_t | S=s, C=c^o , ω^(i)_{t-1} )
        """
        per_model_likelihoods = torch.zeros(self.n_states, (self.n_obs_models + 1))

        for l in range(self.n_states):
            # Existing observation models
            for m in range(self.n_obs_models):
                per_model_likelihoods[l, m] = self.obs_models[l][m].pred_lh(o_t)
            # Novel observation model
            per_model_likelihoods[l, -1] = self.novel_obs_model.pred_lh(o_t)

        # then calculate the likelihood under each context hypothesis for each state
        """
        L_O^(i)( o_t | S=s, C=c, ω^(i)_{t-1} )
        """
        # get the index of the observation model associated with each context hypothesis
        obs_indices = torch.tensor([self.context_model.hypothesis_o(c) for c in range(self.context_model.n_hypotheses)])

        # map the observation model likelihoods to the context hypotheses       
        likelihoods = per_model_likelihoods[:, obs_indices]

        return likelihoods  # [n_states, n_hypotheses]


    def rew_likelihood(self, a_t: int, r_t : float) -> torch.Tensor:
        """
        L_R^(i)( r_t | S=s, C=c, a_t, υ^(i)_{t-1} )

            Given the action a_t, what is the likelihood of the reward r_t under each state l and reward context n?

        returns a tensor of shape (n_states, n_hypotheses) containing the likelihoods for each state and context (including a new context)
        """
        # first calculate under each reward model for each state
        """
        L_R^(i)( r_t | S=s, C=c^r, a_t, υ^(i)_{t-1} )
        """
        per_model_likelihoods = torch.zeros(self.n_states, (self.n_rew_models + 1))

        for l in range(self.n_states):
            for n in range(self.n_rew_models):
                # existing reward models
                per_model_likelihoods[l, n] = self.rew_models[l][n].pred_lh(a_t, r_t)
            # novel reward model
            per_model_likelihoods[l, -1] = self.novel_rew_model.pred_lh(a_t, r_t)

        # then calculate the likelihood under each context hypothesis for each state
        """
        L_R^(i)( r_t | S=s, C=c, a_t, υ^(i)_{t-1} )
        """
        # get the index of the reward model associated with each context hypothesis
        rew_indices = torch.tensor([self.context_model.hypothesis_r(c) for c in range(self.context_model.n_hypotheses)])

        # map the reward model likelihoods to the context hypotheses       
        likelihoods = per_model_likelihoods[:, rew_indices]

        return likelihoods
    

    @property
    def n_obs_models(self):
        """
        Return the number of observation models for each state (i.e. the number of observation contexts)
        """
        return len(self.obs_models[0])

    @property
    def n_rew_models(self):
        """
        Return the number of reward models for each state (i.e. the number of reward contexts)
        """
        return len(self.rew_models[0])

    def create_new_context_models(self, c_o_t, c_r_t):
        """
        Create likelihood models for the new observation and reward contexts if needed
        """

        # Create observation models for new context if needed
        if c_o_t == self.n_obs_models: # equality corresponds to new model since models are indexed from 0 to n-1
            for l, state in enumerate(self.state_space):
                self.obs_models[l].append(self.type_obs(**self.hyp_obs))
        
        # Create reward models for newmodel if needed
        if c_r_t == self.n_rew_models: # equality corresponds to new model since models are indexed from 0 to n-1
            for l, state in enumerate(self.state_space):
                self.rew_models[l].append(self.type_rew(**self.hyp_rew))

    

## TEMPORARYLY SUSPENDED 
# def print_particle_params(particle: Particle):
#     """
#     Print the parameters of the particle's models in a readable format
#     """

#     print("\n> Log weight:", particle.log_weight)

#     # Extract and print the CRP probabilities and sufficient statistics for observation contexts
#     print("\n> Observation Context CRP:")
#     o_prob = particle.cjcrp.CRP_o.probs.detach().numpy()
#     o_prob = np.array2string(o_prob, formatter={'float_kind':lambda x: f"{x: .2f}"})
#     o_ss = particle.cjcrp.CRP_o.counts.detach().numpy()[: particle.cjcrp.CRP_o.n_active_contexts]
#     print("\nObser. Context CRP Probabilities:", o_prob, "Suff Stats:", o_ss)
    
#     # Extract and print observation models
#     print("\n> Observation Models:")
#     for j in range(particle.cjcrp.CRP_o.n_active_contexts):
#         for i, state in enumerate(particle.state_space):
#             # Extract and print format the observation model parameters (df, loc)
#             observation_model_params = particle.obs_models[i][j]._pred_dist_params()
#             df = observation_model_params['df']
#             loc = observation_model_params['loc'].detach().numpy()
#             loc = np.array2string(loc, formatter={'float_kind':lambda x: f"{x: .2f}"})
            
#             print(f"State {state}, Observ Context {j}, Observation Model Params: df: {df}, loc: {loc}")

#     # Extract and print the CRP probabilities and sufficient statistics for reward contexts    
#     print("\n> Reward Context CRP:")
#     r_prob = particle.cjcrp.CRP_r.probs.detach().numpy()
#     r_prob = np.array2string(r_prob, formatter={'float_kind':lambda x: f"{x: .2f}"})
#     r_ss = particle.cjcrp.CRP_r.counts.detach().numpy()[: particle.cjcrp.CRP_r.n_active_contexts]
#     print("\nReward Context CRP Probabilities:", r_prob, "Suff Stats:", r_ss)
    
#     # Extract and print reward models
#     print("\n> Reward Models:")
#     for j in range(particle.cjcrp.CRP_r.n_active_contexts):
#         for i, state in enumerate(particle.state_space):
#             # Extract and print format the reward model mean and sufficient statistics
#             reward_model_mean = particle.rew_models[i][j]._pred_dist_params().detach().numpy()
#             reward_model_mean = np.array2string(reward_model_mean, formatter={'float_kind':lambda x: f"{x: .2f}"})
#             suff_stats_a = particle.rew_models[i][j].post_params()['alpha_n'].detach().numpy()
#             suff_stats_b = particle.rew_models[i][j].post_params()['beta_n'].detach().numpy()
#             suff_stats_a = np.round(suff_stats_a, 2)
#             suff_stats_b = np.round(suff_stats_b, 2)

#             print(f"State {state}, Reward Context {j}, Reward Model Mean: {reward_model_mean}, Suff Stats: {suff_stats_a}, {suff_stats_b}")

   



