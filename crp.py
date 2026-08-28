import torch
import torch.distributions as D


class CRP:
    """
    Class for a Chinese Restaurant Process Prior
    """

    def __init__(self, hyp_alpha):
        self.hyp_alpha = hyp_alpha

        self.counts = torch.zeros(
            100
        )  # this is an artificial upper limit on the number of contexts that is not expected to be reached

        self.n_active_contexts = 0

    @property
    def probs(self):
        """
        Prior probabilities for each category, including a new one
        """
        # returns the probability of each context (including a new one)
        prob = torch.zeros(self.n_active_contexts + 1)
        prob[:-1] = self.counts[: self.n_active_contexts]
        prob[-1] = self.hyp_alpha
        prob = prob / prob.sum()

        return prob

    def update(self, c: int):
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
        counts: torch.Tensor = state["counts"]
        obj.counts = counts.clone()

        if counts.any():
            # Returns the index of the rightmost non-zero element in the count vector, or zero if all elements are 0
            obj.n_active_contexts = counts.nonzero(as_tuple=True)[0].max().item() + 1
        else:
            0

        return obj


class JumpCRP:
    """
    Base class for jump CRP models.

    Handles the common logic for:
      - storing hyperparameters
      - updating CRP statistics only on jumps
      - tracking previous contexts
      - serializing/deserializing state
    """

    def __init__(self, gamma, alpha):
        self.hyp_gamma = gamma
        self.hyp_alpha = alpha

        self.CRP = CRP(hyp_alpha=alpha)
        self.prev_c = 0

    def update(self, h_t, c):
        """Update sufficient statistics and previous context on a jump."""
        if h_t == 1:
            self.CRP.update(c)
            self.prev_c = c

    def to_state(self):
        """Return the minimal state needed to reconstruct the object."""
        return {
            "crp_state": self.CRP.to_state(),
            "prev_c": int(self.prev_c),
        }

    @classmethod
    def from_state(cls, hyp_param: dict, state: dict):
        """Reconstruct a jump CRP from its minimal state."""
        obj = cls(**hyp_param)

        obj.CRP = CRP.from_state(
            obj.hyp_alpha,
            state["crp_state"],
        )
        obj.prev_c = state["prev_c"]

        return obj


class JCRP(JumpCRP):
    """Class for a jump CRP."""

    pass


class CjCRP:
    """
    Class for a coupled jump CRP.

    Contains two independent CRPs, one for each context.
    """

    def __init__(self, gamma, alpha_o, alpha_r):
        self.hyp_gamma = gamma
        self.hyp_alpha_o = alpha_o
        self.hyp_alpha_r = alpha_r

        self.CRP_o = CRP(hyp_alpha=alpha_o)
        self.CRP_r = CRP(hyp_alpha=alpha_r)

        self.prev_c_o = 0
        self.prev_c_r = 0

    def update(self, h_t, c_o, c_r):
        """Update both CRPs and previous contexts on a jump."""
        if h_t == 1:
            self.CRP_o.update(c_o)
            self.CRP_r.update(c_r)

            self.prev_c_o = c_o
            self.prev_c_r = c_r

    def to_state(self):
        """Return the minimal state needed to reconstruct the object."""
        return {
            "crp_o_state": self.CRP_o.to_state(),
            "crp_r_state": self.CRP_r.to_state(),
            "prev_c_o": int(self.prev_c_o),
            "prev_c_r": int(self.prev_c_r),
        }

    @classmethod
    def from_state(cls, hyp_param: dict, state: dict):
        """Reconstruct a coupled jump CRP from its minimal state."""
        cjcrp = cls(**hyp_param)

        cjcrp.CRP_o = CRP.from_state(
            cjcrp.hyp_alpha_o,
            state["crp_o_state"],
        )
        cjcrp.CRP_r = CRP.from_state(
            cjcrp.hyp_alpha_r,
            state["crp_r_state"],
        )

        cjcrp.prev_c_o = state["prev_c_o"]
        cjcrp.prev_c_r = state["prev_c_r"]

        return cjcrp
