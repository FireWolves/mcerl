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
#include "algorithms.hpp"
namespace Env
{

class Environment
{

public:
  Environment(int num_agents, int max_steps, int max_steps_per_agent, int velocity, int sensor_range, int num_rays,
              int min_frontier_pixel, int max_frontier_pixel);

  FrameData step(int agent_id, Action target_index);

  FrameData reset(GridMap env_map, std::vector<Coord> poses);

  Done done() { return is_done_; };

  GridMap env_map() const { return *env_map_; }
  GridMap global_map() const { return *global_map_; }
  GridMap agent_map(int agent_id) const { return *agents_[agent_id].state.map; }

  /** some test function for pybind11 debugging **/

  GridMap test_map_update(GridMap map, GridMap map_to_update, Coord pos, int sensor_range, int num_rays)
  {
    Alg::map_update(std::make_shared<GridMap>(map), &map_to_update, pos, sensor_range, num_rays);
    return map_to_update;
  }
  std::vector<FrontierPoint> test_frontier_detection(GridMap map, int min_pixels, int max_pixels, int sensor_range)
  {
    return Alg::frontier_detection(&map, min_pixels, max_pixels, sensor_range);
  }
  Path test_a_star(GridMap map, Coord start, Coord end) { return Alg::a_star(&map, start, end); }

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
  int min_frontier_pixel_;
  int max_frontier_pixel_;
  int max_steps_per_agent_;
  bool is_done_;
  int tick_;

  void init(GridMap env_map, std::vector<Coord> poses);
  void set_action(int agent_id, Action target_idx);
  FrameData get_frame_data(int agent_id);
  int get_next_act_agent();
  int step_once();

  void reset_state() { step_count_ = 0, is_done_ = false, tick_ = 0; }
};

} // namespace Env
