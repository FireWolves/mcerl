#include "env.hpp"
#include "algorithms.hpp"
#include <iostream>

namespace Env
{

Environment::Environment(int num_agents, int max_steps, int max_steps_per_agent, int velocity, int sensor_range,
                         int num_rays, int min_frontier_pixel, int max_frontier_pixel)
    : num_agents_(num_agents), max_steps_(max_steps), max_steps_per_agent_(max_steps_per_agent), velocity_(velocity),
      sensor_range_(sensor_range), num_rays_(num_rays), min_frontier_pixel_(min_frontier_pixel),
      max_frontier_pixel_(max_frontier_pixel)
{
  agents_.resize(num_agents);
}

void Environment::init(GridMap env_map, std::vector<Coord> poses)
{
  std::cout << "init" << std::endl;
  this->env_map_ = std::make_shared<GridMap>(env_map);
  this->global_map_ = std::make_shared<GridMap>(env_map.rows(), env_map.cols(), UNKNOWN);
  std::cout << "env_map created" << std::endl;
  this->init_poses_ = poses;

  this->step_count_ = 0;
  this->is_done_ = false;
  std::cout << "resetting agents" << std::endl;

  for (int i = 0; i < num_agents_; i++)
    agents_[i].reset(this->env_map_, poses[i], i, max_steps_per_agent_, sensor_range_, num_rays_);
}

std::tuple<Observation, Done, Info> Environment::reset(GridMap env_map, std::vector<Coord> poses)
{
  std::cout << "reset" << std::endl;
  this->init(env_map, poses);
  Observation obs;
  std::cout << "getting poses" << std::endl;
  for (auto &agent : agents_)
    obs.robot_poses.push_back(agent.state.pos);
  Info info;
  info.step_cnt = step_count_;
  step_count_++;
  Done done = false;
  std::cout << "cycling agents" << std::endl;
  // check if any agent requires a new target
  for (auto &agent : agents_)
  {
    std::cout << "checking agent " << agent.info.id << std::endl;
    if (agent.action.target_idx == NONE)
    {
      std::cout << "agent " << agent.info.id << " requires a new target" << std::endl;
      std::cout << "updating map" << std::endl;
      if (agent.state.map.get() == nullptr)
      {
        std::cout << "ERROR: map is nullptr" << std::endl;
      }
      if (agent.state.map.get() == nullptr)
      {
        std::cout << "ERROR: map is nullptr" << std::endl;
      }
      Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
      std::cout << "Map:" << std::endl;
      std::cout << "merge map" << std::endl;
      Alg::map_merge(global_map_, agent.state.map.get());
      std::cout << "detecting frontier" << std::endl;
      auto frontiers =
          Alg::frontier_detection(agent.state.map.get(), min_frontier_pixel_, max_frontier_pixel_, sensor_range_);
      std::cout << "frontiers size: " << frontiers.size() << std::endl;
      agent.state.frontier_points = frontiers;
      obs.frontier_points = frontiers;
      if (frontiers.empty())
      {
        agent.done.done = true;
        done = true;
      }
      info.robot_id = agent.info.id;
      std::cout << "calculating exploration rate" << std::endl;
      info.exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
      info.agent_step_cnt = agent.info.step_count;
      agent.info.step_count++;
      break;
    }
  }
  return std::make_tuple(obs, done, info);
}

std::tuple<Observation, Reward, Done, Info> Environment::step(int agent_id, Action target_index)
{
  std::cout << "step" << std::endl;

  set_action(agent_id, target_index);
  Observation obs;
  Info info;
  info.step_cnt = step_count_;
  step_count_++;
  Done done = false;
  Reward reward;
  reward.exploration_reward = 0;
  reward.time_step = 0;
  // check if any agent requires a new target
  for (auto &agent : agents_)
  {
    if (agent.action.target_idx == NONE)
    {
      std::cout << "agent " << agent.info.id << " requires a new target" << std::endl;
      Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
      auto new_explored_pixels = Alg::calculate_new_explored_pixels(global_map_, agent.state.map.get());
      std::cout << "new explored pixels: " << new_explored_pixels << std::endl;
      if (agent.info.step_count != 0)
      {
        agent.reward.explored_pixels += new_explored_pixels;

        std::cout << "reward.explored_pixels: " << agent.reward.explored_pixels << std::endl;
      }
      reward.exploration_reward = agent.reward.explored_pixels;
      std::cout << "merge map" << std::endl;
      Alg::map_merge(global_map_, agent.state.map.get());
      std::cout << "detecting frontier" << std::endl;
      auto frontiers =
          Alg::frontier_detection(agent.state.map.get(), min_frontier_pixel_, max_frontier_pixel_, sensor_range_);
      std::cout << "frontiers size: " << frontiers.size() << std::endl;
      agent.state.frontier_points = frontiers;
      obs.frontier_points = frontiers;
      reward.exploration_reward = agent.reward.explored_pixels;
      for (auto &i : agents_)
      {
        obs.robot_poses.push_back(i.state.pos);
      }
      if (frontiers.empty())
      {
        agent.done.done = true;
        done = true;
        std::cout << "done" << std::endl;
      }
      info.robot_id = agent.info.id;
      std::cout << "calculating exploration rate" << std::endl;
      info.exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
      info.agent_step_cnt = agent.info.step_count;

      agent.info.step_count++;
      return std::make_tuple(obs, reward, done, info);
    }
  }

  // move the agent
  std::cout << "moving agent" << std::endl;
  for (int i = 0;; ++i)
  {
    for (auto &agent : agents_)
    {
      if (agent.done.done)
      {
        continue;
      }
      if (agent.state.executing_path != nullptr && !agent.state.executing_path->empty())
      {
        auto next_pos = agent.state.executing_path->front();
        agent.state.executing_path->erase(agent.state.executing_path->begin());
        agent.state.pos = next_pos;
        Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
        auto new_explored_pixels = Alg::calculate_new_explored_pixels(global_map_, agent.state.map.get());
        std::cout << "calculating new explored pixels: " << new_explored_pixels << std::endl;
        if (agent.info.step_count != 0)
        {
          agent.reward.explored_pixels += new_explored_pixels;
        }
        Alg::map_merge(global_map_, agent.state.map.get());
      }
      else
      {
        std::cout << "agent " << agent.info.id << " reached target" << std::endl;
        auto frontiers =
            Alg::frontier_detection(agent.state.map.get(), min_frontier_pixel_, max_frontier_pixel_, sensor_range_);
        std::cout << "frontiers size: " << frontiers.size() << std::endl;
        agent.state.frontier_points = frontiers;
        obs.frontier_points = frontiers;
        reward.exploration_reward = agent.reward.explored_pixels;
        std::cout << "reward.exploration_reward: " << reward.exploration_reward << std::endl;
        for (auto &i : agents_)
        {
          obs.robot_poses.push_back(i.state.pos);
        }
        if (frontiers.empty())
        {
          agent.done.done = true;
          done = true;
        }
        info.robot_id = agent.info.id;
        std::cout << "calculating exploration rate" << std::endl;
        info.exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
        info.agent_step_cnt = agent.info.step_count;
        agent.info.step_count++;
        return std::make_tuple(obs, reward, done, info);
      }
    }
    if (std::find_if(agents_.begin(), agents_.end(), [](const auto &x) { return !x.done.done; }) == agents_.end())
    {
      done = true;
      break;
    }
  }
  return std::make_tuple(obs, reward, done, info);
}

void Environment::set_action(int agent_id, Action target_idx)
{

  agents_[agent_id].action.target_idx = target_idx;
  agents_[agent_id].action.target_x = agents_[agent_id].state.frontier_points[target_idx].pos.x;
  agents_[agent_id].action.target_y = agents_[agent_id].state.frontier_points[target_idx].pos.y;
  std::cout << "set_action for " << agent_id << " action index: " << target_idx
            << " pos: " << agents_[agent_id].state.frontier_points[target_idx].pos << std::endl;
  std::cout << "computing path" << std::endl;
  agents_[agent_id].state.executing_path =
      std::make_unique<Path>(Alg::a_star(agents_[agent_id].state.map.get(), agents_[agent_id].state.pos,
                                         agents_[agent_id].state.frontier_points[target_idx].pos));
  std::cout << "path size: " << agents_[agent_id].state.executing_path->size() << std::endl;
  std::cout << "clear frontier points" << std::endl;
  agents_[agent_id].state.frontier_points.clear();
  std::cout << "clearing explored pixels" << std::endl;
  agents_[agent_id].reward.explored_pixels = 0;
}

} // namespace Env