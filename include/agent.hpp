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
#include <iostream>
#include <memory>
namespace Env
{
constexpr int NONE = -1;
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
};

struct AgentInfo
{
  int id;
  int sensor_range;
  int num_rays;
  int step_count;
  int max_steps;
  std::shared_ptr<GridMap> env_map;
};

struct AgentReward
{
  int explored_pixels = 0;
};

struct AgentDone
{
  bool done;
};
struct AgentAction
{
  int target_idx;
  int target_x;
  int target_y;
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
  void reset(std::shared_ptr<GridMap> env_map, Coord pos, int id, int max_steps, int sensor_range, int num_rays)
  {
    std::cout << "reset agent " << id << std::endl;
    info.env_map = env_map;
    info.id = id;
    info.max_steps = max_steps;
    info.sensor_range = sensor_range;
    info.num_rays = num_rays;
    info.step_count = 0;

    state.frontier_points.clear();
    std::cout << "make map" << std::endl;
    state.map = std::make_unique<GridMap>(env_map->width_, env_map->height_, UNKNOWN);
    if (state.map == nullptr)
    {
      std::cout << "map is nullptr" << std::endl;
    }
    state.executing_path = nullptr;
    state.pos = pos;

    action.target_idx = NONE;

    reward.explored_pixels = 0;

    done.done = false;
  }
};

} // namespace Env
