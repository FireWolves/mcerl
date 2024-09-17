from __future__ import annotations

from typing import Any, Callable, Collection

import numpy as np
import numpy.typing as npt

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
        log_level: str = "INFO",
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
        self._env: mcerl.Environment = mcerl.Environment(log_level)

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
        if policy is None:
            policy = random_policy
        trajectories = []
        frame_data = self.reset(**kwargs)
        if self.env_transform is not None:
            frame_data = self.env_transform(frame_data)
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
            frame_data = self.step(frame_data, return_maps=self.return_maps)
            # if the environment is done, we append the last frame data and break
            if self.done() is True:
                trajectories.append(frame_data)
                break
            if (
                self.env_transform is not None
            ):  # transform function should consider  if environment is done,
                frame_data = self.env_transform(frame_data)

        rollouts = split_trajectories(trajectories)
        rollouts = [pad_trajectory(rollout) for rollout in rollouts]
        rollouts = [refine_trajectory(rollout) for rollout in rollouts]
        return rollouts  # noqa:  RET504

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
        self.env_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = (
            env_transform
        )

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
        return data

    def step(
        self,
        frame_data: dict[str, Any],
        *,
        return_maps: bool = False,
    ) -> dict[str, Any]:
        """
        Performs a step in the environment until next agent.

        Args:
            agent (int): The index of the agent.
            action (int): The action to be performed by the agent.

        Returns:
            dict[str, Any]: The data obtained from the environment after the step.

        """
        agent = frame_data["info"]["agent_id"]
        action = frame_data["action"]
        data = self._env.step(agent, action)
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
                    if (
                        0 <= x + i < cols
                        and 0 <= y + j < rows
                        and grid_map[y + j, x + i] != valid_value
                    ):
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
