from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Collection

import numpy as np
import numpy.typing as npt
import torch

import mcerl
from mcerl.utils import (
    pad_trajectory,
    random_policy,
    refine_trajectory,
    split_trajectories,
)

FREE = 255
OCCUPIED = 0
UNKNOWN = 127
X = 1
Y = 0
INVALID_AGENT_ID = -1


class Env:
    """
    Represents an environment for multi-agent exploration.

    Args:
        num_agents (int): The number of agents in the environment.
        max_steps (int): The maximum number of steps in the environment.
        max_steps_per_agent (int): The maximum number of steps per agent.
        velocity (int): The velocity of the agents.
        sensor_range (int): The range of the agent's sensors.
        num_rays (int): The number of rays emitted by the agent's sensors.
        min_frontier_pixel (int): The minimum value of a frontier pixel.
        max_frontier_pixel (int): The maximum value of a frontier pixel.

    Attributes:
        num_agents (int): The number of agents in the environment.
        max_steps (int): The maximum number of steps in the environment.
        max_steps_per_agent (int): The maximum number of steps per agent.
        velocity (int): The velocity of the agents.
        sensor_range (int): The range of the agent's sensors.
        num_rays (int): The number of rays emitted by the agent's sensors.
        min_frontier_pixel (int): The minimum value of a frontier pixel.
        max_frontier_pixel (int): The maximum value of a frontier pixel.
        _last_data (dict[str, Any] | None): The last data obtained from the environment.
        _env (mcerl.Environment): The underlying environment object.
    """

    def __init__(
        self,
        cpp_env_log_level: str,
        cpp_env_log_path: str,
    ) -> None:
        """
        Initializes a new instance of the Env class.
        Args:
            num_agents (int): The number of agents in the environment.
            max_steps (int): The maximum number of steps in the environment.
            max_steps_per_agent (int): The maximum number of steps per agent.
            velocity (int): The velocity of the agents.
            sensor_range (int): The range of the agent's sensors.
            num_rays (int): The number of rays emitted by the agent's sensors.
            min_frontier_pixel (int): The minimum value of a frontier pixel.
            max_frontier_pixel (int): The maximum value of a frontier pixel.

        Returns:
        None

        """
        self._logger = logging.getLogger()
        self._logger.debug("Creating pybind environment object")
        self._env: mcerl.Environment = mcerl.Environment(cpp_env_log_level, cpp_env_log_path)
        self._logger.debug("Pybind environment object created")

    @property
    def last_data(self) -> dict[str, Any]:
        """
        Gets the last data obtained from the environment.

        Returns:
            dict[str, Any]: The last data obtained from the environment.

        Raises:
            ValueError: If no data is available. Call reset() or step() first.

        """

        if self._last_data is None:
            msg = "No data available. Call reset() or step() first"
            raise ValueError(msg)
        return self._last_data

    def rollout(
        self,
        policy: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        # Multithreading is not supported with requires_grad=True
        requires_grad = kwargs.pop("requires_grad", False)
        return_maps = kwargs.get("return_maps", None)
        self._logger.debug(
            "Start rollout with: require grad: %s, return maps: %s",
            requires_grad,
            return_maps,
        )

        if policy is None:
            policy = random_policy
            self._logger.debug("using random policy")
        else:
            self._logger.debug("using network policy")

        trajectories = []
        with torch.no_grad() if not requires_grad else contextlib.nullcontext():
            self._logger.debug("Resetting env")
            frame_data = self.reset(**kwargs)
            self._logger.debug(
                "Env reset, agent id: %s, num of frontiers: %s, agent done: %s, env done: %s",
                self.get_agent_id(frame_data),
                self.get_num_frontiers(frame_data),
                self.get_done(frame_data),
                self.done(),
            )

            if self.env_transform is not None:
                self._logger.debug("Transforming frame data")
                frame_data = self.env_transform(frame_data)
                self._logger.debug("Frame data transformed")

            self._logger.debug("Stepping env")
            while True:
                # if the agent is done, we set the action to 0 to wait other agent to finish
                if frame_data["done"] or len(frame_data["observation"]["frontier_points"]) == 0 or self.done():
                    self._logger.debug(
                        "Agent done: %s or frontier points empty: %s or env done: %s, setting dummy action 0",
                        self.get_done(frame_data),
                        self.get_num_frontiers(frame_data),
                        self.done(),
                    )
                    frame_data["action"] = 0
                else:
                    # otherwise, we use the policy to get the action
                    self._logger.debug("Getting action from policy")
                    frame_data = policy(frame_data)
                    self._logger.debug("Action got: %s", frame_data["action"])

                # append the frame data with action to the trajectory
                self._logger.debug("Adding frame data to trajectory")
                trajectories.append(frame_data)
                self._logger.debug("Frame data added")

                # step the environment
                self._logger.debug("Stepping cpp environment")
                frame_data = self.step(frame_data, return_maps=self.return_maps)
                self._logger.debug(
                    "Cpp env stepped, frame data got agent id: %s, num of frontiers: %s, agent done: %s, env done: %s",
                    self.get_agent_id(frame_data),
                    self.get_num_frontiers(frame_data),
                    self.get_done(frame_data),
                    self.done(),
                )

                # transform function should consider if environment is done,
                if self.env_transform is not None:
                    self._logger.debug("Transforming frame data")
                    frame_data = self.env_transform(frame_data)
                    self._logger.debug("Frame data transformed")

                # if the environment is done, we append the last frame data and break
                if self.done() is True:
                    self._logger.debug("Env done, adding last frame data to trajectory")
                    frame_data["done"] = True
                    trajectories.append(frame_data)
                    break

            # split the trajectories into rollouts
            self._logger.debug("Splitting trajectories")
            rollouts = split_trajectories(trajectories)
            self._logger.debug("Trajectories splitted")

            self._logger.debug("Padding and refining trajectories")
            rollouts = [pad_trajectory(rollout) for rollout in rollouts]
            rollouts = [refine_trajectory(rollout) for rollout in rollouts]
            self._logger.debug("Trajectories padded and refined")

            self._logger.debug("Rollout done")
            return rollouts

    def reset(
        self,
        grid_map: npt.NDArray[np.uint8],
        *,
        agent_poses: Collection[tuple[int, int]] | None = None,
        num_agents: int,
        max_steps: int,
        max_steps_per_agent: int,
        velocity: int,
        sensor_range: int,
        num_rays: int,
        min_frontier_pixel: int,
        max_frontier_pixel: int,
        exploration_threshold: float = 0.95,
        return_maps: bool = False,
        env_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        check_validation: bool = True,
    ) -> dict[str, Any]:
        """
        Resets the environment to its initial state.

        Args:
            grid_map (npt.NDArray[np.uint8]): The grid map representing the environment.
            agent_poses (Collection[tuple[int, int]] | None, optional): The initial poses of the agents. If pos is None, will random sample valid pos for all agents. Defaults to None.

        Returns:
            dict[str, Any]: The initial data obtained from the environment.

        Raises:
            ValueError: If the grid map is not 2D.
            ValueError: If the number of agent poses does not match the number of agents.

        """
        self.num_agents: int = num_agents
        self.max_steps: int = max_steps
        self.max_steps_per_agent: int = max_steps_per_agent
        self.velocity: int = velocity
        self.sensor_range: int = sensor_range
        self.num_rays: int = num_rays
        self.min_frontier_pixel: int = min_frontier_pixel
        self.max_frontier_pixel: int = max_frontier_pixel
        self.exploration_threshold: float = exploration_threshold
        self.return_maps: bool = return_maps
        self._last_data: dict[str, Any] | None = None
        self.env_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = env_transform
        self.check_validation: bool = check_validation

        if grid_map.ndim != 2:
            msg = "Expected 2D grid map"
            raise ValueError(msg)

        if agent_poses is None:
            agent_poses = self.get_valid_spawn_poses(grid_map, self.num_agents)

        if len(agent_poses) != self.num_agents:
            msg = f"Expected {self.num_agents} agent poses, got {len(agent_poses)}"
            raise ValueError(msg)

        data = self._env.reset(
            mcerl.GridMap(grid_map),
            list(agent_poses),
            self.num_agents,
            self.max_steps,
            self.max_steps_per_agent,
            self.velocity,
            self.sensor_range,
            self.num_rays,
            self.min_frontier_pixel,
            self.max_frontier_pixel,
            self.exploration_threshold,
        )

        data = self.unwrap_data(data)
        curr_agent = data["info"]["agent_id"]
        if self.return_maps:
            data["observation"]["global_map"] = self.global_map()
            data["observation"]["agent_map"] = self.agent_map(curr_agent)
        self._logger.debug("Env reset")
        return data

    def step(
        self,
        frame_data: dict[str, Any],
        *,
        return_maps: bool = False,
        set_action: bool = True,
    ) -> dict[str, Any]:
        """
        Performs a step in the environment until next agent.

        Args:
            agent (int): The index of the agent.
            action (int): The action to be performed by the agent.

        Returns:
            dict[str, Any]: The data obtained from the environment after the step.

        """
        self._logger.debug(
            "Cpp env stepped, frame data got agent id: %s, num of frontiers: %s, agent done: %s, env done: %s",
            self.get_agent_id(frame_data),
            self.get_num_frontiers(frame_data),
            self.get_done(frame_data),
            self.done(),
        )
        agent = frame_data["info"]["agent_id"]
        action = frame_data["action"]
        data = self._env.step(agent, action, self.check_validation, set_action)
        data = self.unwrap_data(data)
        curr_agent = data["info"]["agent_id"]
        if return_maps:
            data["observation"]["global_map"] = self.global_map()
            data["observation"]["agent_map"] = self.agent_map(curr_agent)
        return data

    def unwrap_data(self, data: tuple[Any]) -> dict[str, Any]:
        """
        Unwraps the data obtained from the environment.

        Args:
            data (tuple[Any]): The data obtained from the environment.

        Returns:
            dict[str, Any]: The unwrapped data.

        """

        obs, reward, done, info = data  # type: ignore  # noqa: PGH003
        unwrapped_data = {}
        unwrapped_data["info"] = {
            "agent_id": info.agent_id,
            "agent_step_cnt": info.agent_step_cnt,
            "step_cnt": info.step_cnt,
            "agent_exploration_rate": info.agent_exploration_rate,
            "global_exploration_rate": info.global_exploration_rate,
            "delta_time": info.delta_time,
            "agent_explored_pixels": info.agent_explored_pixels,
            "tick": info.tick,
        }
        unwrapped_data["done"] = done
        unwrapped_data["observation"] = {
            "frontier_points": [
                (
                    *frontier_point.pos,
                    frontier_point.unexplored_pixels,
                    frontier_point.distance,
                )
                for frontier_point in obs.frontier_points
            ],
            "pos": obs.agent_poses,
            "target_pos": obs.agent_targets,
        }
        unwrapped_data["reward"] = {
            "exploration_reward": reward.exploration_reward,
            "time_step_reward": reward.time_step_reward,
        }
        self._last_data = unwrapped_data
        return unwrapped_data

    def sample(self) -> int:
        """
        Samples an action from the available actions.

        Returns:
            int: The sampled action.

        """
        action_space = len(self.last_data["observation"]["frontier_points"])
        if action_space > 0:
            rng = np.random.default_rng()
            return rng.integers(action_space).item()  # type: ignore we need get int rather than np.int64
        return 0

    def done(self) -> bool:
        """
        Checks if the environment is done.

        Returns:
            bool: True if the environment is done, False otherwise.

        """
        return self._env.done()

    def random_action(self) -> tuple[int, int]:
        """
        Generates a random action for the agent.

        Returns:
            tuple[int, int]: The random action.

        """
        return (self.last_data["info"]["agent_id"], self.sample())

    def global_map(self) -> npt.NDArray[np.uint8]:
        """
        Gets the global map of the environment.

        Returns:
            npt.NDArray[np.uint8]: The global map.

        """
        return np.array(self._env.global_map(), copy=False)

    def agent_map(self, agent_idx: int) -> npt.NDArray[np.uint8]:
        """
        Gets the map of the specified agent.

        Args:
            agent_idx (int): The index of the agent.

        Returns:
            npt.NDArray[np.uint8]: The map of the agent.

        """
        return np.array(self._env.agent_map(agent_idx), copy=False)

    def get_valid_spawn_poses(
        self,
        grid_map: npt.NDArray[np.uint8],
        num_poses: int,
        valid_radius: int = 3,
    ) -> Collection[tuple[int, int]]:
        """
        Gets valid spawn poses for the agents.

        Args:
            grid_map (npt.NDArray[np.uint8]): The grid map representing the environment.
            num_poses (int): The number of valid poses to generate.

        Returns:
            Collection[tuple[int, int]]: The valid spawn poses.

        """

        def check_around(x, y, grid_map, radius=3, valid_value=255):
            """
            check if the area around the point is valid
            """
            rows = grid_map.shape[0]
            cols = grid_map.shape[1]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if 0 <= x + i < cols and 0 <= y + j < rows and grid_map[y + j, x + i] != valid_value:
                        return False
            return True

        rows = grid_map.shape[0]
        cols = grid_map.shape[1]
        valid_poses = []
        rng = np.random.default_rng()
        while len(valid_poses) < num_poses:
            y = rng.integers(rows).item()  # type: ignore  # noqa: PGH003
            x = rng.integers(cols).item()  # type: ignore  # noqa: PGH003
            if check_around(x, y, grid_map, radius=valid_radius):
                valid_poses.append((x, y))
        return valid_poses

    def get_agent_id(self, frame_data: dict[str, Any]) -> int:
        return frame_data["info"]["agent_id"]

    def get_num_frontiers(self, frame_data: dict[str, Any]) -> int:
        return len(frame_data["observation"]["frontier_points"])

    def get_done(self, frame_data: dict[str, Any]) -> bool:
        return frame_data["done"]

    def agent_path(self, agent_idx: int) -> list[tuple[int, int]]:
        return self._env.agent_path(agent_idx)

    def agent_pos(self, agent_idx: int) -> tuple[int, int]:
        return self._env.agent_pos(agent_idx)

    def agent_target(self, agent_idx: int) -> tuple[int, int]:
        return self._env.agent_target(agent_idx)

    def agent_step_cnt(self, agent_idx: int) -> int:
        return self._env.agent_step_cnt(agent_idx)

    def step_cnt(self) -> int:
        return self._env.step_cnt()

    def tick(self) -> int:
        return self._env.tick()

    def eval(
        self,
        policy: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        **kwargs,
    ):
        kwargs.pop("return_maps", None)
        kwargs.pop("requires_grad", None)
        eval_trajectories_data = []
        self._logger.debug("Start eval ")

        self._logger.debug("using %s policy", "random" if policy is None else "network")
        if policy is None:
            policy = random_policy

        trajectories = []
        with torch.no_grad():
            self._logger.debug("Resetting env")
            frame_data = self.reset(**kwargs)

            self._logger.debug("Stepping env")
            while True:
                if self.env_transform is not None:
                    self._logger.debug("Transforming frame data")
                    frame_data = self.env_transform(frame_data)
                    self._logger.debug("Frame data transformed")

                # if the agent is done, we set the action to 0 to wait other agent to finish
                if frame_data["done"] or len(frame_data["observation"]["frontier_points"]) == 0 or self.done():
                    self._logger.debug(
                        "Agent done: %s or frontier points empty: %s or env done: %s, setting dummy action 0",
                        self.get_done(frame_data),
                        self.get_num_frontiers(frame_data),
                        self.done(),
                    )
                    frame_data["action"] = 0
                else:
                    # otherwise, we use the policy to get the action
                    self._logger.debug("Getting action from policy")
                    frame_data = policy(frame_data,training=False)
                    self._logger.debug("Action got: %s", frame_data["action"])
                eval_trajectories_data.append(self.get_eval_data(self.get_agent_id(frame_data)))

                # append the frame data with action to the trajectory
                self._logger.debug("Adding frame data to trajectory")
                trajectories.append(frame_data)
                self._logger.debug("Frame data added")

                # step the environment
                self._logger.debug("Stepping cpp environment")
                next_act_agent = self._env.get_next_act_agent()
                action = frame_data["action"]
                act_agent_id = frame_data["info"]["agent_id"]
                if not isinstance(action, int):
                    action = action.item()
                self._env.set_action(act_agent_id, action)
                while next_act_agent == INVALID_AGENT_ID and not self.done():
                    next_act_agent = self._env.step_eval()
                    for i in range(self.num_agents):
                        eval_data = self.get_eval_data(i)
                        eval_trajectories_data.append(eval_data)
                frame_data = self.step(frame_data, return_maps=False, set_action=False)

                # if the environment is done, we append the last frame data and break
                if self.done() is True:
                    self._logger.debug("Env done, adding last frame data to trajectory")
                    frame_data["done"] = True
                    if self.env_transform is not None:
                        # transform function should consider if environment is done,
                        self._logger.debug("Transforming frame data")
                        frame_data = self.env_transform(frame_data)
                        self._logger.debug("Frame data transformed")
                    trajectories.append(frame_data)
                    break

            # split the trajectories into rollouts
            self._logger.debug("Splitting trajectories")
            rollouts = split_trajectories(trajectories)
            self._logger.debug("Trajectories splitted")

            self._logger.debug("Padding and refining trajectories")
            rollouts = [pad_trajectory(rollout) for rollout in rollouts]
            rollouts = [refine_trajectory(rollout) for rollout in rollouts]
            self._logger.debug("Trajectories padded and refined")

            self._logger.debug("Rollout done")
            return rollouts, eval_trajectories_data

    def get_eval_data(self, agent_id: int) -> dict[str, Any]:
        self._logger.debug("Gathering evaluation data")
        return {
            "agent_id": agent_id,
            "agent_pos": self.agent_pos(agent_id),
            "agent_target": self.agent_target(agent_id),
            "agent_path": self.agent_path(agent_id),
            "agent_map": self.agent_map(agent_id),
            "global_map": self.global_map(),
            "agent_step_cnt": self.agent_step_cnt(agent_id),
            "step_cnt": self.step_cnt(),
            "tick": self.tick(),
        }
