/**
 * @file common.hpp
 * @author zhaoth
 * @brief  common data structure
 * @version 0.1
 * @date 2024-08-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <vector>

namespace Env
{

constexpr uint8_t FREE = 255;    /**< The value of a free cell in the environment. */
constexpr uint8_t OCCUPIED = 0;  /**< The value of an occupied cell in the environment. */
constexpr uint8_t UNKNOWN = 127; /**< The value of an unknown cell in the environment. */
/**
 * @brief Alias for a coordinate in the environment.
 */
using Coord = cv::Point;

/**
 * @brief Alias for a path in the environment.
 */
using Path = std::vector<Coord>;

/**
 * @brief Represents a frontier point in the environment.
 */
struct FrontierPoint
{
  Coord pos;             /**< The position of the frontier point. */
  int unexplored_pixels; /**< The number of unexplored pixels around the frontier point. */
  int distance;          /**< The distance to the agent. */
};

/**
 * @brief Represents an observation in the environment.
 */
struct Observation
{
  std::vector<FrontierPoint> frontier_points; /**< The list of frontier points in the observation. */
  std::vector<Coord> agent_poses;             /**< The list of agent poses in the observation. */
  std::vector<Coord> agent_targets;           /**< The list of agent targets in the observation. */
};

/**
 * @brief Represents additional information about the environment.
 */
struct Info
{
  int agent_id; /**< The ID of the agent. */
  int step_cnt; /**< The step count in the environment. */
  int agent_step_cnt;
  float global_exploration_rate; /**< The exploration rate of the environment. */
  float agent_exploration_rate;  /**< The exploration rate of the agent. */
  int delta_time;                /**< The time difference in the environment. */
  int agent_explored_pixels;     /**< The number of explored pixels in the environment. */
};

/**
 * @brief Represents the reward in the environment.
 */
struct Reward
{
  int exploration_reward; /**< The exploration reward in the environment. */
  int time_step_reward;   /**< The time step in the environment. */
};

/**
 * @brief Alias for the done flag in the environment.
 */
using Done = bool;

/**
 * @brief Alias for an action in the environment.
 */
using Action = int;
using FrameData = std::tuple<Observation, Reward, Done, Info>;

} // namespace Env

/**
 * @brief opencv point formatter for fmt library, copied from github
 *
 */
template <> struct fmt::formatter<cv::Point>
{
  constexpr auto parse(format_parse_context &ctx) { return ctx.end(); }

  template <typename Context> auto format(const cv::Point &p, Context &ctx)
  {
    return format_to(ctx.out(), "[{}, {}]", p.x, p.y);
  }
};