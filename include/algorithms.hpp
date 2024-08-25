#pragma once

#include "common.hpp"
#include "grid_map.hpp"
#include <memory>
namespace Alg
{

Env::Path a_star(Env::GridMap *map, Env::Coord start, Env::Coord end);

void map_update(std::shared_ptr<Env::GridMap> map_env, Env::GridMap *map_update, Env::Coord pos, int sensor_range,
                int num_rays);
std::vector<Env::FrontierPoint> frontier_detection(Env::GridMap *map);
float exploration_rate(std::shared_ptr<Env::GridMap> map_env, Env::GridMap *map_exploration);
void map_merge(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *map_update);
int calculate_new_explored_pixels(std::shared_ptr<Env::GridMap> map_env, Env::GridMap *map_update);
} // namespace Alg