from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Collection

import numpy as np
import tensordict
import torch
import tqdm
from tensordict import LazyStackedTensorDict

import mcerl
from mcerl.env import Env


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


def single_env_rollout(
    env: Env,
    grid_map: np.ndarray,
    policy: Callable[[dict[str, Any]], dict[str, Any]] = random_policy,
    agent_poses: Collection[tuple[int, int]] | None = None,
    *,
    env_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    return_maps: bool = True,
) -> list[list[dict[str, Any]]]:
    """
    Perform a single environment rollout.
    after the rollout, we split the trajectory into agent-wise trajectories and pad them, then refine them.
    Args:
        env (Env): The environment object.
        grid_map (np.ndarray): The grid map.
        agent_poses (Collection[tuple[int, int]]): The initial positions of the agents. Defaults to None. If None, the agents are placed randomly.
        policy (Callable[[dict], int]): The policy function that takes in an observation and returns an action index. Defaults to random_policy.
        return_maps (bool): Whether to return the maps in the trajectory. Defaults to True.
    Returns:
        List[List[dict[str,Any]]]: A list of trajectories.
    """
    with torch.no_grad():
        trajectories = []
        frame_data = env.reset(grid_map, agent_poses, return_maps=return_maps)
        if env_transform is not None:
            frame_data = env_transform(frame_data)
        while True:
            # if the agent is done, we set the action to 0 to wait other agent to finish
            if (
                frame_data["done"]
                or len(frame_data["observation"]["frontier_points"]) == 0
            ):
                frame_data["action"] = 0
            else:
                # otherwise, we use the policy to get the action
                frame_data = policy(frame_data)
            # append the frame data with action to the trajectory
            trajectories.append(frame_data)
            # step the environment
            frame_data = env.step(frame_data, return_maps=return_maps)
            # if the environment is done, we append the last frame data and break
            if env.done() is True:
                trajectories.append(frame_data)
                break
            # if the environment is not done, and agent is not done, we transform the frame data
            if env_transform is None or frame_data["done"]:
                continue
            frame_data = env_transform(frame_data)
        rollouts = split_trajectories(trajectories)
        rollouts = [pad_trajectory(rollout) for rollout in rollouts]
        rollouts = [refine_trajectory(rollout) for rollout in rollouts]
    return rollouts  # noqa:  RET504


def multi_threaded_rollout(
    env: Callable[..., mcerl.Environment],
    policy: Callable[[dict[str, Any]], dict[str, Any]],
    grid_map: np.ndarray,
    agent_poses: Collection[tuple[int, int]] | None = None,
    *,
    return_maps: bool = False,
    env_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    num_threads: int,
    epochs: int,
) -> list[list[dict[str, Any]]]:
    """
    Perform a multi-threaded rollout.
    Args:
        env (Callable[..., mcerl.Environment]): The environment instance function.
        policy (Callable[[dict[str, Any]], dict[str, Any]]): The policy function that takes in frame data with observation and returns with action and state value (if critic is used).
        grid_map (np.ndarray): The grid map.
        agent_poses (Collection[tuple[int, int]]): The initial positions of the agents, if None, the agents are placed randomly.
        num_threads (int): The number of threads to use.
        epochs (int): The number of epochs to run.
    Returns:
        List[List[dict[str,Any]]]: A list of trajectories.
    """

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(epochs):
            future = executor.submit(
                single_env_rollout,
                env(),
                grid_map.copy(),
                policy,
                agent_poses,
                return_maps=return_maps,
                env_transform=env_transform,
            )
            futures.append(future)
        rollouts = []
        for future in tqdm.tqdm(futures, desc="Rollouts"):
            rollout = future.result()
            rollouts.extend(rollout)
        return rollouts


def exploration_reward_rescale(
    trajectory: list[dict[str, Any]],
    max_value: float,
) -> list[dict[str, Any]]:
    """
    Standardize the exploration reward to [0,1].
    Args:
        trajectory (list[dict[str, Any]]): The trajectory contains list of frame data.
    Returns:
        dict[str, Any]: The updated frame data.
    """
    for i in range(len(trajectory)):
        trajectory[i]["next"]["reward"]["exploration_reward"] = (
            trajectory[i]["next"]["reward"]["exploration_reward"] / max_value
        )
    return trajectory


def delta_time_reward_standardize(
    trajectory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Standardize the delta time reward to [0,1] (approximately).
    Args:
        trajectory (list[dict[str, Any]]): The trajectory contains list of frame data.
    Returns:
        dict[str, Any]: The updated frame data.
    """
    max_value = max(
        [frame["next"]["reward"]["time_step_reward"] for frame in trajectory]
    )
    min_value = min(
        [frame["next"]["reward"]["time_step_reward"] for frame in trajectory]
    )
    for i in range(len(trajectory)):
        trajectory[i]["next"]["reward"]["time_step_reward"] = -(
            trajectory[i]["next"]["reward"]["time_step_reward"] - min_value
        ) / (max_value - min_value)
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
