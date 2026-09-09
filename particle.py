from typing import TypedDict
import torch
import numpy as np
import torch.distributions as D
from models_obs_rew import ConjugateModel

from models_context import ContextModel
from utils_sample import normalise, bern_sample, cat_sample, cat2D_sample


"""
Particle = ParticleParams + ParticleState
"""
class ParticleParams(TypedDict):
    """
    TypedDict for the parameters of a Particle object.
    """
    type_context: type[ContextModel]
    type_obs :    type[ConjugateModel]
    type_rew :    type[ConjugateModel]
    hyp_context:  dict
    hyp_obs:      dict
    hyp_rew:      dict    

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
    def __init__(self, particle_params: ParticleParams):
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

        # Coupled jump CRP prior
        self.hyp_gamma = self.hyp_context["gamma"]
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
    def from_state(cls, particle_params: ParticleParams, state : ParticleState):
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

    

    def create_new_context_models(self, j_t, c_o_t, c_r_t):
        """
        Create likelihood models for the new observation and reward contexts if needed
        """

        # New models only required if jump occurs
        if j_t == 1: 

            # Create observation models for new context if needed
            if c_o_t == self.context_model.n_contexts_o: # equality corresponds to new context since contexts are indexed from 0 to n-1
                for l, state in enumerate(self.state_space):
                    self.obs_models[l].append(self.type_obs(**self.hyp_obs))
            
            # Create reward models for newcontext if needed
            if c_r_t == self.context_model.n_contexts_r:
                for l, state in enumerate(self.state_space):
                    self.rew_models[l].append(self.type_rew(**self.hyp_rew))



    def obs_likelihood(self, o_t : torch.Tensor) -> torch.Tensor:
        """
        P(o_t | s_t=l, c^o_t=m)

            What is the likelihood of the observation o_t under each state l and observation context m?

        returns a tensor of shape (n_states, n_contexts_o + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(self.n_states, (self.context_model.n_contexts_o + 1))

        for l in range(self.n_states):
            for m in range(self.context_model.n_contexts_o):
                """
                P(o_t | s_t=l, c^o_t=m) = ∫ P(o_t | ω_l,m) P(ω_l,m) dω_l,m
                """
                likelihoods[l, m] = self.obs_models[l][m].pred_lh(o_t)
            """
            P(o_t | s_t=l, c^o_t=new) = ∫ P(o_t | ω_l,new) P(ω_l,new) dω_l,new
            """
            likelihoods[l, -1] = self.novel_obs_model.pred_lh(o_t)

        return likelihoods

    def rew_likelihood(self, a_t: int, r_t : float) -> torch.Tensor:
        """
        P(r_t | s_t=l, c^r_t=n, a_t)

            Given the action a_t, what is the likelihood of the reward r_t under each state l and reward context n?

        returns a tensor of shape (n_states, n_contexts_r + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(self.n_states, (self.context_model.n_contexts_r + 1))

        for l in range(self.n_states):
            for n in range(self.context_model.n_contexts_r):
                """
                P(r_t | s_t=l, c^r_t=n, a_t) = ∫ P(r_t | υ_l,n,a) P(υ_l,n,a) dυ_l,n,a        
                """
                likelihoods[l, n] = self.rew_models[l][n].pred_lh(a_t, r_t)
            """
            P(r_t | s_t=l, c^r_t=new, a_t) = ∫ P(r_t | υ_l,new,a) P(υ_l,new,a) dυ_l,new,a   
            """
            likelihoods[l, -1] = self.novel_rew_model.pred_lh(a_t, r_t)

        return likelihoods




    def before_action(self, o_t: torch.Tensor):
        """
        Compute
        P(S_t, C^o_t, C^r_t, J_t | o_t, h_{t-1})
        """

        # Calculate observation likelihoods for each state and observation context
        """
        P(o_t | S_t=l, C^o_t=m) 
        """
        self._obs_lh = self.obs_likelihood(o_t)


        # Compute belief over states conditional on jump event and observation
        """
        P(S_t=l | o_t, J_t=0, ...) 
            ∝ P(o_t | S_t=l, C^o_t=c_o_{t-1}) P(S_t=l)
        """
        self.state_belief_stay = normalise(torch.einsum('l,l->l', self._obs_lh[:, self.context_model.prev_c_o], self.state_probs))
        if self.context_model.n_contexts_o == 0: # overwrite for the first trial when there are no observation contexts yet
            self.state_belief_stay = torch.zeros_like(self.state_probs)
        """
        P(S_t=l | o_t, J_t=1, ...) 
            ∝ Σ_m P(o_t | S_t=l, C^o_t=m) P(C^o_t=m | θ^o) P(S_t=l)
        """
        self.state_belief_jump = normalise(torch.einsum('lm,m,l->l', self._obs_lh, self.context_model.probs_c_o, self.state_probs))


        # Compute marginal likelihoods (evidence) for each branch of the mixture
        """
        P(o_t | J_t=0) 
            = Σ_l P(o_t | S_t=l, C^o_t=c_o_{t-1}) P(S_t=l)
        """
        stay_evidence = torch.einsum('l,l->', self.state_probs, self._obs_lh[:, self.context_model.prev_c_o])
        if self.context_model.n_contexts_o == 0:
            stay_evidence = torch.tensor(0.0)
        """
        P(o_t | J_t=1) 
            = Σ_l Σ_m P(o_t | S_t=l, C^o_t=m) P(C^o_t=m | θ^o) P(S_t=l)
        """
        jump_evidence = torch.einsum("lm,m,l->",  self._obs_lh, self.context_model.probs_c_o, self.state_probs)

        # Total evidence for the observation under the mixture model of jump vs stay
        """
        P(o_t | ...) 
            = P(J_t=0) P(o_t | J_t=0, ...) + P(J_t=1) P(o_t | J_t=1, ...)
        """
        mixture_evidence = ((1.0 - self.hyp_gamma)) * stay_evidence + ((self.hyp_gamma) * jump_evidence)
        self.log_weight += torch.log(mixture_evidence + 1e-30)

        """
        P(J_t=1 | o_t)
        """
        self.p_jump_o = torch.clamp((self.hyp_gamma * jump_evidence) / (mixture_evidence + 1e-30), 0.0, 1.0)

        


    def sample_action(self):
        """
        Thompson sampling using the temporary belief induced by o_t
        """
        """
        j_hat ~ P(J_t | o_t)
        """
        j_hat = bern_sample(self.p_jump_o)

        # Sample state conditional on that branch
        """
        s_hat ~ P(S_t | o_t, J_t=j_hat)
        """
        if j_hat == 0:
            s_hat = cat_sample(self.state_belief_stay)
        else:
            s_hat = cat_sample(self.state_belief_jump)

        # Sample reward context conditional on that branch
        """
        c_r_hat ~ P(C^r_t | o_t, J_t=j_hat)
        """
        if j_hat == 0:
            c_r_hat = self.context_model.prev_c_r
        else:
            c_r_hat = cat_sample(self.context_model.probs_c_r)

        # -----------------------------------------------
        #   Thompson sampling under provisional latents
        # -----------------------------------------------
        # Sample reward parameters for (s_hat, c_r_hat)
        """
        υ_hat ~ P(ϒ | s_hat, c_r_hat)
        """
        if c_r_hat == self.context_model.n_contexts_r:
            predicted_rewards = self.novel_rew_model.sample_post_dist() # special case for novel context
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
        P(S_t, C^o_t, C^r_t, H_t | h_t)
        """
    
        """
        P(r_t | S_t, C^r_t, a_t)
        """
        self._rew_lh = self.rew_likelihood(a_t, r_t)  # (n_states, n_contexts_r+1)

        # ------------------------------------
        #   Sample jump, contexts, and state 
        # ------------------------------------    
        # 1) Sample Jump
        """
        P(o_t, r_t | J_t=0, a_t) 
            ∝ Σ_l P(r_t | S_t=l, c_r_{t-1}, a_t) P(S_t=l | o_t, J_t=0, ...)
        """
        stay_evidence = torch.einsum('l,l->', self._rew_lh[:, self.context_model.prev_c_r], self.state_belief_stay)
        if self.context_model.n_contexts_r == 0: # overwrite for the first trial when there are no reward contexts yet
            stay_evidence = torch.tensor(0.0)

        """
        P(J_t=1 | o_t, a_t, r_t) 
            ∝ Σ_n Σ_l P(r_t | S_t=l, C^r_t=n, a_t) P(C^r_t=n | θ^r) P(S_t=l | o_t, J_t=1, ...) 
        """
        jump_evidence = torch.einsum('ln,n,l->',  self._rew_lh, self.context_model.probs_c_r, self.state_belief_jump)

        # predictive reward evidence 
        """
        P(r_t | o_t, a_t, ...) 
            = P(J_t = 0 | o_t, ...) P(r_t | J_t = 0, a_t, o_t, ...) + P(J_t = 1 | o_t, ...) P(r_t | J_t = 1, o_t, a_t, ...) 
        """
        mixture_evidence = (1.0 - self.p_jump_o) * stay_evidence + self.p_jump_o * jump_evidence
        self.log_weight += torch.log(mixture_evidence + 1e-30)

        # posterior jump probability given o and r
        """
        P(J_t = 1 | o_t, a_t, r_t, ...) 
            ∝ P(J_t = 1 | o_t, ...) P(r_t | J_t = 1, o_t, a_t, ...)
        """
        p_jump_ora = torch.clamp((self.p_jump_o * jump_evidence) / (mixture_evidence + 1e-30), 0.0, 1.0)
        j_t = bern_sample(p_jump_ora)
        
        # 2) Sample contexts given jump
        if j_t == 0:
            """
            P(C^o_t = l, C^r_t = m | j_t=0, ...)
                = δ(C^o_t = c_o_{t-1}) δ(C^r_t = c_r_{t-1})
            """
            c_o_t, c_r_t = int(self.context_model.prev_c_o), int(self.context_model.prev_c_r)
        else:
            """
            P(C^o_t = m, C^r_t = n | j_t=1, ...) 
                ∝ Σ_l P(S_t=l) P(o_t | S_t=l, C^o_t=m) P(r_t | S_t=l, C^r_t=n, a_t) P(C^o_t=m | θ^o) P(C^r_t=n | θ^r)
            """
            context_probs = torch.einsum('l,lm,ln,m,n->mn', self.state_probs, self._obs_lh, self._rew_lh, self.context_model.probs_c_o, self.context_model.probs_c_r)
            c_o_t, c_r_t = cat2D_sample(context_probs)
        
        # 3) sample state given committed contexts:
        """
        P(S_t=l | o_t, r_t, a_t, c^o_t, c^r_t) 
            ∝ P(r_t | S_t=l, c^r_t, a_t) P(o_t | S_t=l, c^o_t) P(S_t=l)
        """
        state_prob = torch.einsum('l,l,l->l', self.state_probs, self._rew_lh[:, c_r_t], self._obs_lh[:, c_o_t])
        s_t = cat_sample(state_prob)



        # ---------------------
        #   Update parameters 
        # ---------------------    
        # 1) Create new likelihood models if neeeded
        self.create_new_context_models(j_t, c_o_t, c_r_t)
        
        # 2) Update observation model for the sampled context
        self.obs_models[s_t][c_o_t].update(o_t)
        
        # 3) Update reward model for the sampled context
        self.rew_models[s_t][c_r_t].update(a_t, r_t)
        
        # 4) Update CRP visit counts and previous contexts
        self.context_model.update(j_t, c_o_t, c_r_t)

        return s_t, c_o_t, c_r_t, j_t

    

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

   



