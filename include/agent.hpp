/**
 * @file agent.hpp
 * @author zhaoth
 * @brief agent data structure
 * @version 0.1
 * @date 2024-08-24
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "common.hpp"
#include "grid_map.hpp"
#include <memory>
#include <spdlog/spdlog.h>
namespace Env
{
/**
 * @brief NONE value
 *
 */
constexpr int NONE = -1;
constexpr int MAX_FAILED_ATTEMPTS = 5;
/**
 * @brief base class for state
 *
 */
struct AgentState
{
  Coord pos;
  std::unique_ptr<GridMap> map;
  std::unique_ptr<Path> executing_path;
  std::vector<FrontierPoint> frontier_points;
  void reset()
  {
    executing_path = nullptr;
    frontier_points.clear();
  }
  void reset(Coord pos, int map_width, int map_height)
  {
    reset();
    this->pos = pos;
    this->map = std::make_unique<GridMap>(map_width, map_height, UNKNOWN);
  }
};

struct AgentInfo
{
  int id;
  int sensor_range;
  int num_rays;
  int step_count;
  int max_steps;
  int delta_time;
  int failed_attempts;
  float exploration_rate;
  std::shared_ptr<GridMap> env_map;
  void reset() { delta_time = 0, exploration_rate = 0.0; }
  void reset(int id, int sensor_range, int num_rays, int max_steps)
  {
    reset();
    this->id = id;
    this->sensor_range = sensor_range;
    this->num_rays = num_rays;
    this->max_steps = max_steps;
    this->step_count = 0;
    this->failed_attempts = 0;
  }
  void reset(int id, int sensor_range, int num_rays, int max_steps, std::shared_ptr<GridMap> env_map)
  {
    reset(id, sensor_range, num_rays, max_steps);
    this->env_map = env_map;
  }
};

struct AgentReward
{
  int explored_pixels = 0;
  void reset() { explored_pixels = 0; }
};

struct AgentDone
{
  bool done = false;
  void reset() { done = false; }
};
struct AgentAction
{
  int target_idx;
  Coord target_pos;

  void reset() { target_idx = NONE; }
};
class Agent
{
public:
  AgentAction action;
  AgentState state;
  AgentInfo info;
  AgentReward reward;
  AgentDone done;
  Agent() = default;
  void reset()
  {
    state.reset();
    info.reset();
    reward.reset();
    done.reset();
    action.reset();
  }
  void reset(std::shared_ptr<GridMap> env_map, Coord pos, int id, int max_steps, int sensor_range, int num_rays)
  {
    spdlog::debug("resetting agent {} at pos: {}", id, pos);
    state.reset(pos, env_map->width(), env_map->height());
    info.reset(id, sensor_range, num_rays, max_steps, env_map);
    reward.reset();
    done.reset();
    action.reset();
  }
};

} // namespace Env
