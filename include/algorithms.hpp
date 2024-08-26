#pragma once

#include "common.hpp"
#include "grid_map.hpp"
#include <memory>
namespace Alg
{

int calculate_surrounding_unexplored_pixels(Env::GridMap *exploration_map, Env::Coord pos, int range);
void map_update(std::shared_ptr<Env::GridMap> env_map, Env::GridMap *exploration_map, Env::Coord pos, int sensor_range,
                int num_rays);
std::vector<Env::FrontierPoint> frontier_detection(Env::GridMap *exploration_map, int min_pixels, int max_pixels,
                                                   int sensor_range);
Env::Path a_star(Env::GridMap *exploration_map, Env::Coord start, Env::Coord end);
void map_merge(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *agent_map);
int calculate_new_explored_pixels(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *agent_map);
float exploration_rate(std::shared_ptr<Env::GridMap> env_map, Env::GridMap *exploration_map);

} // namespace Alg