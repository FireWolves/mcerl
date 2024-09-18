from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Collection

import numpy as np
import tensordict
from tensordict import LazyStackedTensorDict

# from mcerl.env import Env


def split_trajectories(trajectories) -> list[list[dict[str, Any]]]:
    """
    split trajectory into agent-wise trajectories
    """

    agent_trajectories = {}
    for frame_data in trajectories:
        agent_id = frame_data["info"]["agent_id"]
        if agent_id not in agent_trajectories:
            agent_trajectories[agent_id] = []
        agent_trajectories[agent_id].append(frame_data)
    return list(agent_trajectories.values())


def pad_trajectory(trajectory) -> list[dict[str, Any]]:
    """
    TODO: I messy this up, need to refactor this
    In this environment, we won't get an observation when done is True.
    However, we need to pad the trajectories to stack them.
    use T-1's observation to pad T, it's ok because we never use this state (normally).
    we also delete those states after done for waiting for remaining agents to finish.
    """
    if len(trajectory) < 2:
        return trajectory
    trajectory_out = []
    for i in range(len(trajectory)):
        trajectory_out.append(trajectory[i])
        if trajectory[i]["done"]:
            break
    trajectory_out[-1]["done"] = True
    trajectory_out[-1]["value"] = 0.0
    return trajectory_out


def refine_trajectory(
    trajectory, exclude_keys: Collection[str] = ("action", "log_prob")
) -> list[dict[str, Any]]:
    """transform the trajectory
    TODO: we need to use hard code to mark the keys to exclude, maybe we can use a more elegant way to do this
    (Obs_k,Info_k,Done_k Action_k, Reward_k-1)
    to
    (Obs_k, Info_k,Done_k, Action_k,
    next(Reward_k, Obs_k+1,Info_k+1,Done_k+1)
    )"""
    refined_trajectory = []
    for i in range(len(trajectory) - 1):
        refined_trajectory.append(
            deepcopy(
                {
                    **{
                        key: value
                        for key, value in trajectory[i].items()
                        if key != "reward"
                    },
                    "next": {
                        key: value
                        for key, value in trajectory[i + 1].items()
                        if key not in exclude_keys
                    },
                }
            )
        )

    return refined_trajectory


def stack_trajectory(trajectory):
    """
    stack trajectory to tensordict
    """
    return LazyStackedTensorDict.maybe_dense_stack(
        [tensordict.TensorDict(frame_data) for frame_data in trajectory]
    )


def random_policy(frame_data: dict[str, Any]) -> dict[str, Any]:
    """
    random policy for the agents

    """
    action_space = len(frame_data["observation"]["frontier_points"])
    if action_space > 0:
        rng = np.random.default_rng()
        action = rng.integers(action_space).item()  # type: ignore  # noqa: PGH003
        frame_data["action"] = action
    else:
        frame_data["action"] = 0
    return frame_data


def exploration_reward_rescale(
    trajectory: list[dict[str, Any]],
    a: float,
) -> list[dict[str, Any]]:
    """
    Standardize the exploration reward to [0,1].
    Args:
        trajectory (list[dict[str, Any]]): The trajectory contains list of frame data.
    Returns:
        dict[str, Any]: The updated frame data.
    """
    for i in range(len(trajectory)):
        trajectory[i]["next"]["reward"]["exploration_reward"] = math.tanh(
            trajectory[i]["next"]["reward"]["exploration_reward"] / a
        )
    return trajectory


def delta_time_reward_standardize(
    trajectory: list[dict[str, Any]], *, sigma: float, b: int, x_star: float
) -> list[dict[str, Any]]:
    """
    Standardize the delta time reward to [0,1] (approximately).
    Args:
        trajectory (list[dict[str, Any]]): The trajectory contains list of frame data.
    Returns:
        dict[str, Any]: The updated frame data.
    """
    # max_value = max(
    #     [frame["next"]["reward"]["time_step_reward"] for frame in trajectory]
    # )
    # min_value = min(
    #     [frame["next"]["reward"]["time_step_reward"] for frame in trajectory]
    # )
    total_value = sum(
        [frame["next"]["reward"]["time_step_reward"] for frame in trajectory]
    )
    for i in range(len(trajectory)):
        trajectory[i]["next"]["reward"]["time_step_reward"] = 1 / (
            ((trajectory[i]["next"]["reward"]["time_step_reward"] - x_star) / sigma)
            ** (2 * b)
            + 1
        )
        trajectory[i]["info"].update({"total_time_step": total_value})
    return trajectory


def reward_sum(
    trajectory: list[dict[str, Any]], gamma: float = 0.95
) -> list[dict[str, Any]]:
    """
    Sum the rewards in the trajectory.
    Args:
        trajectory (list[dict[str, Any]]): The trajectory contains list of frame data.
    Returns:
        list[dict[str, Any]]: The trajectory contains list of frame data with the rewards summed and episode return added.
    """
    reward_to_go = 0
    for i in range(len(trajectory) - 1, -1, -1):
        trajectory[i]["next"]["reward"].update(
            {"total_reward": sum(trajectory[i]["next"]["reward"].values())}
        )
        reward_to_go = (
            trajectory[i]["next"]["reward"]["total_reward"] + gamma * reward_to_go
        )
        trajectory[i].update({"reward_to_go": reward_to_go})
    return trajectory
