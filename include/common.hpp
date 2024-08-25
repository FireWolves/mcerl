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
};

/**
 * @brief Represents an observation in the environment.
 */
struct Observation
{
  std::vector<FrontierPoint> frontier_points; /**< The list of frontier points in the observation. */
  std::vector<Coord> robot_poses;             /**< The list of robot poses in the observation. */
};

/**
 * @brief Represents additional information about the environment.
 */
struct Info
{
  int robot_id; /**< The ID of the robot. */
  int step_cnt; /**< The step count in the environment. */
  int agent_step_cnt;
  float exploration_rate; /**< The exploration rate of the environment. */
  // int delta_time;         /**< The time difference in the environment. */
};

/**
 * @brief Represents the reward in the environment.
 */
struct Reward
{
  int exploration_reward; /**< The exploration reward in the environment. */
  int time_step;          /**< The time step in the environment. */
};

/**
 * @brief Alias for the done flag in the environment.
 */
using Done = bool;

/**
 * @brief Alias for an action in the environment.
 */
using Action = int;
} // namespace Env