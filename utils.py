import numpy as np
import torch


def niceprint(x: np.ndarray | torch.Tensor, decimals: int) -> str:
    """
    Format a numpy array or torch tensor to a string with the given number of decimal places for each element
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    return np.array2string(x, formatter={"float_kind": lambda x: f"{x: .{decimals}f}"})
