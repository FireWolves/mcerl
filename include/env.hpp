/**
 * @file env.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-08-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "agent.hpp"
namespace Env
{

class Environment
{

public:
  Environment(int num_agents, int max_steps, int max_steps_per_agent, int velocity);

  void init(GridMap env_map, std::vector<Coord> poses);

  std::tuple<Observation, Reward, Done, Info> step(int agent_id, Action target_index);

  std::tuple<Observation, Done, Info> reset(GridMap env_map, std::vector<Coord> poses);

  Done done()
  {
    return is_done_;
  };
  void test_grid_map();
  GridMap env_map() const
  {
    return *env_map_;
  }
  GridMap global_map()
  {
    return *global_map_;
  }
  GridMap local_map(int agent_id)
  {
    return *agents_[agent_id].state.map;
  }

private:
  std::vector<Agent> agents_;
  std::shared_ptr<GridMap> env_map_;
  std::shared_ptr<GridMap> global_map_;
  std::vector<Coord> init_poses_;
  int velocity_; // pixels per step
  int sensor_range_;
  int num_rays_;
  int num_agents_;
  int max_steps_;
  int step_count_;
  int max_steps_per_agent_;
  bool is_done_ = false;
  void set_action(int agent_id, Action target_idx);
};

} // namespace Env
