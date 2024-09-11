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
                       int threshold, bool check_reachability = false, Env::GridMap *exploration_map = nullptr,
                       Env::Coord agent_pos = {0, 0});
int count_pixels(Env::GridMap *map, uint8_t value, bool exclude_value);
} // namespace Alg