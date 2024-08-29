from __future__ import annotations

from typing import Any, Collection

import numpy as np
import numpy.typing as npt

import mcerl

FREE = 255
OCCUPIED = 0
UNKNOWN = 127


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
        *,
        num_agents: int,
        max_steps: int,
        max_steps_per_agent: int,
        velocity: int,
        sensor_range: int,
        num_rays: int,
        min_frontier_pixel: int,
        max_frontier_pixel: int,
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
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.max_steps_per_agent = max_steps_per_agent
        self.velocity = velocity
        self.sensor_range = sensor_range
        self.num_rays = num_rays
        self.min_frontier_pixel = min_frontier_pixel
        self.max_frontier_pixel = max_frontier_pixel

        self._last_data: dict[str, Any] | None = None

        self._env: mcerl.Environment = mcerl.Environment(
            num_agents,
            max_steps,
            max_steps_per_agent,
            velocity,
            sensor_range,
            num_rays,
            min_frontier_pixel,
            max_frontier_pixel,
        )

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

    def reset(
        self,
        grid_map: npt.NDArray[np.uint8],
        agent_poses: Collection[tuple[int, int]] | None = None,
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
        if grid_map.ndim != 2:
            msg = "Expected 2D grid map"
            raise ValueError(msg)

        if agent_poses is None:
            agent_poses = self.get_valid_spawn_poses(grid_map, self.num_agents)

        if len(agent_poses) != self.num_agents:
            msg = f"Expected {self.num_agents} agent poses, got {len(agent_poses)}"
            raise ValueError(msg)

        data = self._env.reset(mcerl.GridMap(grid_map), list(agent_poses))

        return self.unwrap_data(data)

    def step(self, agent: int, action: int) -> dict[str, Any]:
        """
        Performs a step in the environment until next agent.

        Args:
            agent (int): The index of the agent.
            action (int): The action to be performed by the agent.

        Returns:
            dict[str, Any]: The data obtained from the environment after the step.

        """
        data = self._env.step(agent, action)
        return self.unwrap_data(data)

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
        }
        unwrapped_data["done"] = done
        unwrapped_data["observation"] = {
            "frontier_points": [
                (*frontier_point.pos, frontier_point.unexplored_pixels)
                for frontier_point in obs.frontier_points
            ],
            "pos": obs.agent_poses,
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
        self, grid_map: npt.NDArray[np.uint8], num_poses: int
    ) -> Collection[tuple[int, int]]:
        """
        Gets valid spawn poses for the agents.

        Args:
            grid_map (npt.NDArray[np.uint8]): The grid map representing the environment.
            num_poses (int): The number of valid poses to generate.

        Returns:
            Collection[tuple[int, int]]: The valid spawn poses.

        """
        rows = grid_map.shape[0]
        cols = grid_map.shape[1]
        valid_poses = []
        rng = np.random.default_rng()
        while len(valid_poses) < num_poses:
            row = rng.integers(rows).item()
            col = rng.integers(cols).item()
            if grid_map[row, col] == 255:
                valid_poses.append((row, col))
        return valid_poses
