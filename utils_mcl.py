import numpy as np
import torch
from pathlib import Path


def extract_context_params(folder, episode_idx=0):
    folder = Path(folder)
    active_means = np.load(folder / "active_means")
    T, B, _ = active_means.shape
    active_means = active_means[:, episode_idx, :]
    active_sigmas = np.loadtxt(folder / "active_sigmas")[:, episode_idx]
    correct_actions = np.load(folder / "correct_actions")[:, episode_idx, :]
    contexts = np.loadtxt(folder / "ctxs")
    class_ids = np.loadtxt(folder / "class_ids")[:, episode_idx]
    states = np.loadtxt(folder / "class_ids")[:, episode_idx]
    stimuli = np.load(folder / "stimuli")[:, episode_idx]

    # Collect observations for each (context, class)
    observations = {}

    for t in range(T):
        context = int(contexts[t])
        class_id = int(class_ids[t])
        key = (context, class_id)

        if key not in observations:
            observations[key] = {
                "means": [],
                "sigmas": [],
                "p_rews": [],
            }

        observations[key]["means"].append(active_means[t])
        observations[key]["sigmas"].append(active_sigmas[t])
        observations[key]["p_rews"].append(correct_actions[t])

    # ------------------------------------------------------------------
    # Build context_o_params
    # ------------------------------------------------------------------

    context_o_params = {}

    for (context, class_id), obs in sorted(observations.items()):
        means = np.asarray(obs["means"])
        sigmas = np.asarray(obs["sigmas"])

        mean = means.mean(axis=0)
        sigma = sigmas.mean()

        covariance = np.eye(2) * sigma**2

        context_o_params.setdefault(context, {})[class_id] = {
            "loc": torch.tensor(mean, dtype=torch.float32),
            "cov": torch.tensor(covariance, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Build context_r_params
    # ------------------------------------------------------------------

    context_r_params = {}

    for (context, class_id), obs in sorted(observations.items()):
        p_rews = np.asarray(obs["p_rews"])

        p_rew = p_rews[0]

        if not np.allclose(p_rews, p_rew):
            raise ValueError(
                f"Inconsistent p_rew for context={context}, "
                f"class={class_id}: {p_rews}"
            )

        if not np.isclose(p_rew.sum(), 1.0):
            raise ValueError(
                f"p_rew is not one-hot for context={context}, "
                f"class={class_id}: {p_rew}"
            )

        context_r_params.setdefault(context, {})[class_id] = {
            "p_rew": torch.tensor(p_rew, dtype=torch.float32)
        }

    return contexts, states, stimuli, context_r_params
