from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Collection

import numpy as np
import tensordict
import tqdm
from tensordict import LazyStackedTensorDict

import mcerl
from mcerl.env import Env


def split_trajectories(trajectories):
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


def pad_trajectory(trajectory) -> list:
    """
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
            trajectory_out[-1]["observation"] = trajectory_out[-2]["observation"]
            break
    return trajectory_out


def refine_trajectory(trajectory):
    """transform the trajectory
    (Obs_k,Info_k,Done_k Action_k, Reward_k-1)
    to
    (Obs_k, Info_k,Done_k, Action_k,
    next(Reward_k, Obs_k+1,Info_k+1,Done_k+1)
    )"""
    refined_trajectory = []
    for i in range(len(trajectory) - 1):
        refined_trajectory.append(
            {
                "observation": trajectory[i]["observation"],
                "info": trajectory[i]["info"],
                "done": trajectory[i]["done"],
                "action": trajectory[i]["action"],
                "next": {
                    "reward": trajectory[i + 1]["reward"],
                    "observation": trajectory[i + 1]["observation"],
                    "info": trajectory[i + 1]["info"],
                    "done": trajectory[i + 1]["done"],
                },
            }
        )
    return refined_trajectory


def stack_trajectory(trajectory):
    """
    stack trajectory to tensordict
    """
    return LazyStackedTensorDict.maybe_dense_stack(
        [tensordict.TensorDict(frame_data) for frame_data in trajectory]
    )


def single_env_rollout(
    env: Env,
    grid_map: np.ndarray,
    policy: Callable[[dict], int],
    agent_poses: Collection[tuple[int, int]] | None = None,
):
    """
    Perform a single environment rollout.
    Args:
        env (Env): The environment object.
        grid_map (np.ndarray): The grid map.
        agent_poses (Collection[tuple[int, int]]): The initial positions of the agents.
        policy (Callable[[dict], int]): The policy function that takes in an observation and returns an action index.
    Returns:
        List[Tensordict]: A list of stacked trajectories.
    """
    trajectories = []
    frame_data = env.reset(grid_map, agent_poses)
    trajectories.append(frame_data)
    while True:
        agent_id = frame_data["info"]["agent_id"]
        action_index = policy(frame_data["observation"])
        frame_data["action"] = action_index
        frame_data = env.step(agent_id, action_index)
        trajectories.append(frame_data)
        if env.done() is True:
            break
    rollouts = split_trajectories(trajectories)
    rollouts = [pad_trajectory(rollout) for rollout in rollouts]
    rollouts = [refine_trajectory(rollout) for rollout in rollouts]
    return [stack_trajectory(rollout) for rollout in rollouts]


def multi_threaded_rollout(
    env: Callable[..., mcerl.Environment],
    policy: Callable[[dict], int],
    grid_map: np.ndarray,
    agent_poses: Collection[tuple[int, int]] | None = None,
    *,
    num_threads: int,
    epochs: int,
):
    """
    Perform a multi-threaded rollout.
    Args:
        env (Callable[..., mcerl.Environment]): The environment class.
        policy (Callable[[dict], int]): The policy function that takes in an observation and returns an action index.
        grid_map (np.ndarray): The grid map.
        agent_poses (Collection[tuple[int, int]]): The initial positions of the agents.
        num_threads (int): The number of threads to use.
        epochs (int): The number of epochs to run.
    Returns:
        List[Tensordict]: A list of stacked trajectories.
    """

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in tqdm.tqdm(range(epochs), desc="Epochs"):
            future = executor.submit(
                single_env_rollout, env(), grid_map.copy(), policy, agent_poses
            )
            futures.append(future)
        rollouts = []
        for future in tqdm.tqdm(futures, desc="Rollouts"):
            rollout = future.result()
            rollouts.extend(rollout)
        return rollouts
