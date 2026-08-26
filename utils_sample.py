import torch
import torch.distributions as D


def normalise(probs: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """
    Normalise a tensor of probabilities to sum to 1, with numerical stability.

    :param torch.Tensor probs: tensor of probabilities (not necessarily normalized)
    :param float eps: small constant to prevent division by zero
    :return torch.Tensor: normalized probabilities
    """
    total = torch.sum(probs) + eps
    return probs / total


def bern_sample(prob: float | torch.Tensor) -> int:
    """
    Sample from a Bernoulli distribution with given probability.

    :param float | torch.Tensor prob: probability of success (between 0 and 1)
    :return int: sampled value (0 or 1)
    """
    return int(D.Bernoulli(probs=prob).sample().item())


def cat_sample(probs: torch.Tensor) -> int:
    """
    Sample from a categorical distribution defined by the given probabilities.

    :param torch.Tensor probs: tensor of probabilities (not necessarily normalized)
    :return int: sampled category index
    """
    return int(D.Categorical(probs).sample().item())


class Categorical2D:
    """
    Class for representing a 2D categorical distribution [e.g. a joint P(X,Y)].
    """

    def __init__(self, probs: torch.Tensor):
        """
        :param torch.Tensor probs: 2D tensor of probabilities (not necessarily normalized)
        """
        assert probs.dim() == 2, "probs must be a 2D tensor"
        self.dim1, self.dim2 = probs.shape
        self.probs2d = probs / torch.sum(probs)
        self.probs = probs.flatten()

    def sample(self) -> tuple[int, int]:
        """
        Sample from the 2D categorical distribution and return the corresponding indices (i, j).
        """
        idx = D.Categorical(probs=self.probs).sample().item()
        i = idx // self.dim2
        j = idx % self.dim2

        return int(i), int(j)


def cat2D_sample(probs: torch.Tensor) -> tuple[int, int]:
    """
    Sample from a 2D categorical distribution defined by the given probabilities.

    :param torch.Tensor probs: 2D tensor of probabilities
    :return tuple[int, int]: sampled indices (i, j)
    """
    return Categorical2D(probs).sample()
