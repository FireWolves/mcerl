#pragma once

#include "common.hpp"
#include "grid_map.hpp"
#include <memory>
namespace Alg
{

int calculate_surrounding_unexplored_pixels(Env::GridMap *exploration_map, Env::Coord pos, int range);
void map_update(std::shared_ptr<Env::GridMap> env_map, Env::GridMap *exploration_map, Env::Coord pos, int sensor_range,
                int num_rays, int expand_pixels = 2);
std::vector<Env::FrontierPoint> frontier_detection(Env::GridMap *exploration_map, int min_pixels, int max_pixels,
                                                   int sensor_range);
Env::Path a_star(Env::GridMap *exploration_map, Env::Coord start, Env::Coord end, int tolerance_range = 5);
void map_merge(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *agent_map);
int calculate_valid_explored_pixels(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *agent_map);
float exploration_rate(std::shared_ptr<Env::GridMap> env_map, Env::GridMap *exploration_map);
bool is_frontier_valid(std::shared_ptr<Env::GridMap> global_map, Env::FrontierPoint &frontier_point, int radius,
                       int threshold);
int count_pixels(Env::GridMap *map, uint8_t value, bool exclude_value);
int calculate_path_distance(const Env::Path &path);
std::vector<cv::Point> ray_trace(const cv::Mat &static_mat, Env::Coord pos,
                                 const std::vector<Env::Coord> &circle_end_points, int expand_pixels);
Env::GridMap map_update_with_polygon(const cv::Mat &static_mat, cv::Mat &mat_to_update, Env::Coord pos,
                                     std::vector<cv::Point> polygon);
std::vector<cv::Point> calculate_circle_points_with_random_offset(const std::vector<cv::Point2d> &unit_circles,
                                                                  Env::Coord pos, int radius, double random_offset_min,
                                                                  double random_offset_max, int rows, int cols);
std::vector<cv::Point2d> calculate_unit_circled_points(int num_points);
int calculate_valid_unexplored_pixels(std::shared_ptr<Env::GridMap> map, Env::Coord coord, int ray_range,
                                      const std::vector<cv::Point2d> &unit_circles, int expand_pixels);
std::vector<cv::Point> calculate_circle_points(const std::vector<cv::Point2d> &unit_circles, Env::Coord pos, int radius,
                                               int rows, int cols);
} // namespace Alg