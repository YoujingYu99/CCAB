from typing import TypedDict, Literal

import torch
import torch.distributions as D


class ContextHypothesis(TypedDict):
    """
    A context hypothesis represents a possible combination of a jump, an observation model, and a reward model.
    """
    jump : Literal[0, 1]
    obs_model : int
    rew_model : int

class ContextModel():
    """
    General interface for a context model. This is an abstract class that should be subclassed to implement specific context models.
    The methods here provide type hints and documentation for the expected behavior of context models.
    """

    @property
    def n_hypotheses(self) -> int:
        return len(self.context_hypotheses)

    @property
    def hypothesis_probs(self) -> torch.Tensor:
        """
        P_C(C_t = c | xi_{t-1}) for all currently possible c.
        Shape: [n_hypotheses]
        """
        raise NotImplementedError

    def hypothesis_o(self, c: int) -> int:
        """
        Return the index of the observation model associated with context hypothesis c.
        """
        return self.context_hypotheses[c]["obs_model"]

    def hypothesis_r(self, c: int) -> int:
        """
        Return the index of the reward model associated with context hypothesis c.
        """
        return self.context_hypotheses[c]["rew_model"]

    def hypothesis_jump(self, c: int) -> int:
        """
        Return the jump indicator (0 or 1) associated with context hypothesis c.
        """
        return self.context_hypotheses[c]["jump"]

    def update(self, c_t: int):
        raise NotImplementedError

    @classmethod
    def from_state(cls, hyp_param: dict, state: dict):
        raise NotImplementedError

    def to_state(self) -> dict:
            raise NotImplementedError

    


class CRP():
    """
    Class for a Chinese Restaurant Process Prior
    """
    def __init__(self, hyp_alpha):
        self.hyp_alpha = hyp_alpha
        
        self.counts = torch.zeros(100) # this is an artificial upper limit on the number of contexts that is not expected to be reached

        self.n_active_contexts = 0

    @property
    def probs(self):
        """
        Prior probabilities for each category, including a new one
        """
        # returns the probability of each context (including a new one)
        prob = torch.zeros(self.n_active_contexts+1)
        prob[:-1] = self.counts[:self.n_active_contexts]
        prob[-1] = self.hyp_alpha
        prob = prob / prob.sum()
        
        return prob

    def update(self, c:int):
        """
        Update the counts with a single context assignment
        """
        self.counts[c] += 1

        if c == self.n_active_contexts:
            self.n_active_contexts += 1        
            
    def sample(self) -> int:
        """
        Sample from the CRP prior
        """
        return D.Categorical(probs=self.probs).sample().item()

    def to_state(self):
        """
        Minimal state representation needed to reconstruct the CRP object
        """
        return {"counts": self.counts}

    @classmethod
    def from_state(cls, hyp_alpha, state):
        """
        Reconstruct a CRP object from its minimal state representation
        """
        obj = cls(hyp_alpha=hyp_alpha)
        counts : torch.Tensor = state["counts"]
        obj.counts = counts.clone()
        
        if counts.any():
            # Returns the index of the rightmost non-zero element in the count vector, or zero if all elements are 0
            obj.n_active_contexts = counts.nonzero(as_tuple=True)[0].max().item() + 1
        else: 
            0
        
        return obj


class jCRP(ContextModel):
    """
    Class for a jump CRP
    """
    def __init__(self, gamma: float, alpha: float):
        self.hyp_gamma = gamma
        self.hyp_alpha = alpha

        self.CRP = CRP(hyp_alpha=alpha)

        # Due to the jump mixture, we need to keep track of the previous context
        self.prev_c = 0

        # Fill the list of context hypotheses based on the current state of the CRP
        self.fill_context_hypotheses()

    def fill_context_hypotheses(self):
        """
        Fill the list of context hypotheses based on the current state of the CRP.
        A single context determines both the observation model and reward model.
        """
        self.context_hypotheses = []

        # If there are no active contexts, we can only have a jump to a new context
        if self.CRP.n_active_contexts == 0:
            self.context_hypotheses.append({
                "jump":      1, 
                "obs_model": 0, 
                "rew_model": 0
                })

        # Otherwise
        else:
            # J = 0: exactly one possible hypothesis
            self.context_hypotheses.append(
                {
                    "jump":      0,
                    "obs_model": self.prev_c,
                    "rew_model": self.prev_c,
                })

            # J = 1: one hypothesis for each possible CRP context
            for c in range(self.CRP.n_active_contexts + 1):
                self.context_hypotheses.append(
                    {
                        "jump":      1,
                        "obs_model": c,
                        "rew_model": c,
                    })

    @property
    def hypothesis_probs(self):

        # If there is only one hypothesis, it must be the jump-to-new-context hypothesis, so its probability is 1
        if len(self.context_hypotheses) == 1:
            probs = torch.ones(1)

        # Otherwise, compute probabilities based on the CRP prior and the jump probability
        else:
            probs = torch.zeros(self.n_hypotheses)

            # Hypothesis 0 is always the unique stay hypothesis
            probs[0] = 1.0 - self.hyp_gamma

            # Remaining hypotheses are jumps
            for i, hyp in enumerate(self.context_hypotheses[1:], start=1):
                c = hyp["obs_model"]

                # The same context determines both observation and reward models
                probs[i] = self.hyp_gamma * self.CRP.probs[c]

        return probs

    def update(self, c_t: int):
        """
        Wrapper function for updating the underlying CRP.
        """
        # Extract the jump and shared context from the current hypothesis
        hyp = self.context_hypotheses[c_t]
        j_t = hyp["jump"]
        c_t = hyp["obs_model"]

        # Sufficient statistics are only updated when a jump occurs
        if j_t == 1:
            # Update the single shared context CRP
            self.CRP.update(c_t)

            # Update previous context
            self.prev_c = c_t

            self.fill_context_hypotheses()  # Refill the context hypotheses after the update

    def to_state(self):
        """
        Minimal state needed to reconstruct the object.
        """
        return {
            "crp_state": self.CRP.to_state(),
            "prev_c":    int(self.prev_c),
        }

    @classmethod
    def from_state(cls, hyp_param: dict, state: dict):
        """
        Reconstruct a jCRP object from its minimal state representation.
        """
        jcrp = cls(**hyp_param)

        jcrp.CRP = CRP.from_state(jcrp.hyp_alpha, state["crp_state"])
        jcrp.prev_c = state["prev_c"]

        jcrp.fill_context_hypotheses()  # Refill the context hypotheses after restoring the state

        return jcrp


class CjCRP(ContextModel):
    """
    Class for a coupled jump CRP
    """
    def __init__(self, gamma : float,  alpha_o : float, alpha_r : float):
        self.hyp_gamma   = gamma
        self.hyp_alpha_o = alpha_o
        self.hyp_alpha_r = alpha_r

        # Set up individual CRPs for 
        self.CRP_o = CRP(hyp_alpha=alpha_o)
        self.CRP_r = CRP(hyp_alpha=alpha_r)

        # Due to the jump mixture, we need to keep track of the previous contexts
        self.prev_c_o = 0
        self.prev_c_r = 0

        # Fill the list of context hypotheses based on the current state of the CRPs
        self.fill_context_hypotheses()

    def fill_context_hypotheses(self):
        """
        Fill the list of context hypotheses based on the current state of the CRPs.
        Each hypothesis is a combination of a jump (0 or 1), an observation model, and a reward model.
        """
        self.context_hypotheses = []

        # If there are no active contexts, we can only have a jump to a new context
        if self.CRP_o.n_active_contexts == 0 and self.CRP_r.n_active_contexts == 0:
            self.context_hypotheses.append({
                "jump":      1,
                "obs_model": 0,
                "rew_model": 0
            })

        # Otherwise
        else:
            # J = 0: exactly one possible hypothesis
            self.context_hypotheses.append({
                "jump":      0,
                "obs_model": self.prev_c_o,
                "rew_model": self.prev_c_r,
            })

            # J = 1: Cartesian product of CRP possibilities
            for c_o in range(self.CRP_o.n_active_contexts + 1):
                for c_r in range(self.CRP_r.n_active_contexts + 1):
                    self.context_hypotheses.append({
                        "jump":    1,
                        "obs_model": c_o,
                        "rew_model": c_r,
                    })

    @property
    def hypothesis_probs(self):

        # If there is only one hypothesis, it must be the stay hypothesis, so its probability is 1
        if len(self.context_hypotheses) == 1:
            probs = torch.ones(1)

        # Otherwise, we compute the probabilities for each hypothesis based on the CRP priors and the jump probability
        else:
            probs = torch.zeros(self.n_hypotheses)

            # Hypothesis 1 is always the unique stay hypothesis
            probs[0] = 1.0 - self.hyp_gamma

            # Remaining hypotheses 1, 2, 3 ... are jumps
            for i, hyp in enumerate(self.context_hypotheses[1:], start=1):
                c_o = hyp["obs_model"]
                c_r = hyp["rew_model"]

                probs[i] = (self.hyp_gamma * self.CRP_o.probs[c_o] * self.CRP_r.probs[c_r])

        return probs
     
    def update(self, c_t: int):
        """
        Wrapper function that combines updates to the underlying CRPs.
        """
        # Extract the jump, obs_model, and rew_model from the current context hypothesis
        hyp = self.context_hypotheses[c_t]
        j_t = hyp["jump"]
        c_o_t = hyp["obs_model"]
        c_r_t = hyp["rew_model"]

        # Sufficient statistics are only updated when a jump occurs
        if j_t == 1:
            # Update counts
            self.CRP_o.update(c_o_t)
            self.CRP_r.update(c_r_t)
            
            # Update previous context
            self.prev_c_o = c_o_t
            self.prev_c_r = c_r_t

            self.fill_context_hypotheses()  # Refill the context hypotheses after the update

    def to_state(self):
        """
        Minimal state needed to reconstruct the object. This is a combination of counts and the previous state for each CRP.
        """
        return {
            "crp_o_state":  self.CRP_o.to_state(),
            "crp_r_state":  self.CRP_r.to_state(),
            "prev_c_o":     int(self.prev_c_o),
            "prev_c_r":     int(self.prev_c_r)
        }


    
    @classmethod
    def from_state(cls, hyp_param:dict, state:dict):
        """
        Reconstruct a cjCRP object from its minimal state representation
        """
        cjcrp = cls(**hyp_param)
        
        cjcrp.CRP_o = CRP.from_state(cjcrp.hyp_alpha_o, state["crp_o_state"])
        cjcrp.CRP_r = CRP.from_state(cjcrp.hyp_alpha_r, state["crp_r_state"])

        cjcrp.prev_c_o = state["prev_c_o"]
        cjcrp.prev_c_r = state["prev_c_r"]

        cjcrp.fill_context_hypotheses()  # Refill the context hypotheses after restoring the state

        return cjcrp