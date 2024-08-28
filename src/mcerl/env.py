from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import mcerl


class Env:
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
        if self._last_data is None:
            msg = "No data available. Call reset() or step() first"
            raise ValueError(msg)
        return self._last_data

    def reset(
        self, grid_map: npt.NDArray[np.uint8], agent_poses: list[tuple[int, int]]
    ) -> dict[str, Any]:
        if len(agent_poses) != self.num_agents:
            msg = f"Expected {self.num_agents} agent poses, got {len(agent_poses)}"
            raise ValueError(msg)
        if grid_map.ndim != 2:
            msg = "Expected 2D grid map"
            raise ValueError(msg)
        data = self._env.reset(mcerl.GridMap(grid_map), agent_poses)
        return self.unwrap_data(data)

    def step(self, agent: int, action: int) -> dict[str, Any]:
        data = self._env.step(agent, action)
        return self.unwrap_data(data)

    def unwrap_data(self, data: tuple[Any]) -> dict[str, Any]:
        obs, reward, done, info = data  # type: ignore  # noqa: PGH003
        unwrapped_data = {}
        unwrapped_data["robot_id"] = info.robot_id
        unwrapped_data["step_cnt"] = info.agent_step_cnt
        unwrapped_data["total_step_cnt"] = info.step_cnt
        unwrapped_data["exploration_rate"] = info.exploration_rate
        unwrapped_data["reward"] = {
            "exploration": reward.exploration_reward,
            "time_elapsed": info.delta_time,
        }
        unwrapped_data["done"] = done
        unwrapped_data["obs"] = {
            "frontier_points": [
                (*frontier_point.pos, frontier_point.unexplored_pixels)
                for frontier_point in obs.frontier_points
            ],
            "pos": obs.robot_poses,
        }
        self._last_data = unwrapped_data
        return unwrapped_data

    def sample(self) -> int:
        action_space = len(self.last_data["obs"]["frontier_points"])
        if action_space > 0:
            rng = np.random.default_rng()
            return rng.integers(action_space)
        return 0

    def done(self) -> bool:
        return self._env.done()

    def random_action(self) -> tuple[int, int]:
        return (self.last_data["robot_id"], self.sample())
