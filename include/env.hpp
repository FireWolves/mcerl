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
  Environment(const std::string log_level, const std::string log_file);

  FrameData step(int agent_id, Action target_index);

  FrameData reset(GridMap env_map, std::vector<Coord> poses, int num_agents, int max_steps, int max_steps_per_agent,
                  int velocity, int sensor_range, int num_rays, int min_frontier_pixel, int max_frontier_pixel,
                  float exploration_threshold);

  Done done() { return is_done_; };

  GridMap env_map() const { return *env_map_; }
  GridMap global_map() const { return *global_map_; }
  GridMap agent_map(int agent_id) const { return *agents_[agent_id].state.map; }

  /** some test function for pybind11 debugging **/

  std::tuple<std::vector<Coord>, std::vector<Coord>, std::vector<Coord>, GridMap, GridMap> test_map_update(
      GridMap map, GridMap map_to_update, Coord pos, int sensor_range, int num_rays, int expand_pixels)
  {
    auto &&static_mat = cv::Mat(map.rows(), map.cols(), CV_8UC1, map.data());
    auto &&map_to_update_mat = cv::Mat(map_to_update.rows(), map_to_update.cols(), CV_8UC1, map_to_update.data());
    auto &&basic_end_points = Alg::calculate_unit_circled_points(num_rays);
    auto &&circle_end_points = Alg::calculate_circle_points_with_random_offset(basic_end_points, pos, sensor_range, 0,
                                                                               M_PI / num_rays, map.rows(), map.cols());
    auto &&circle_end_points_with_polygon = Alg::ray_trace(static_mat, pos, circle_end_points, 5);
    auto &&roi_map = Alg::map_update_with_polygon(static_mat, map_to_update_mat, pos, circle_end_points_with_polygon);
    std::vector<Coord> basic_end_points_i;
    for (auto &i : basic_end_points)
    {
      basic_end_points_i.push_back(Coord(static_cast<int>(i.x * 10000), static_cast<int>(i.y * 10000)));
    }
    return {basic_end_points_i, circle_end_points, circle_end_points_with_polygon, map_to_update, roi_map};
  }
  std::vector<FrontierPoint> test_frontier_detection(GridMap map, int min_pixels, int max_pixels, int sensor_range)
  {
    return Alg::frontier_detection(&map, min_pixels, max_pixels, sensor_range);
  }
  Path test_a_star(GridMap map, Coord start, Coord end) { return Alg::a_star(&map, start, end); }
  auto test_xy_coord(GridMap map, Coord coord) { return map(coord.x, coord.y); }
  auto test_xy_cv_mat(GridMap map, Coord coord)
  {
    auto mat = cv::Mat(map.rows(), map.cols(), CV_8UC1, map.data());
    return mat.at<uint8_t>(coord);
  }

  void init(GridMap env_map, std::vector<Coord> poses, int num_agents, int max_steps, int max_steps_per_agent,
            int velocity, int sensor_range, int num_rays, int min_frontier_pixel, int max_frontier_pixel,
            float exploration_threshold);

private:
  std::vector<Agent> agents_;
  std::shared_ptr<GridMap> env_map_;
  std::shared_ptr<GridMap> global_map_;
  std::vector<Coord> init_poses_;
  std::vector<cv::Point2d> unit_circle_end_points_;
  int velocity_; // pixels per step
  int sensor_range_;
  int num_rays_;
  int num_agents_;
  int max_steps_;
  int step_count_;
  int min_frontier_pixel_;
  int max_frontier_pixel_;
  int max_steps_per_agent_;
  float exploration_threshold_;
  bool is_done_;
  int tick_;
  int ray_cast_random_offs_min_;
  int ray_cast_random_offs_max_;
  int ray_cast_expand_pixels_;

  void set_action(int agent_id, Action target_idx);
  FrameData get_frame_data(int agent_id);
  int get_next_act_agent();
  int step_once();

  void reset_state() { step_count_ = 0, is_done_ = false, tick_ = 0; }
};

} // namespace Env
