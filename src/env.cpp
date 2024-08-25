#include "env.hpp"
#include "algorithms.hpp"

namespace Env
{

Environment::Environment(int num_agents, int max_steps, int max_steps_per_agent, int velocity, int sensor_range,
                         int num_rays)
    : num_agents_(num_agents), max_steps_(max_steps), max_steps_per_agent_(max_steps_per_agent), velocity_(velocity),
      sensor_range_(sensor_range), num_rays_(num_rays)
{
  agents_.resize(num_agents);
}

void Environment::init(GridMap env_map, std::vector<Coord> poses)
{
  this->env_map_ = std::make_shared<GridMap>(env_map);
  this->init_poses_ = poses;
  this->step_count_ = 0;
  this->is_done_ = false;
  for (int i = 0; i < num_agents_; i++)
    agents_[i].reset(this->env_map_, poses[i], i, max_steps_per_agent_, , 8);
}

std::tuple<Observation, Done, Info> Environment::reset(GridMap env_map, std::vector<Coord> poses)
{
  this->init(env_map, poses);
  Observation obs;
  for (auto &agent : agents_)
    obs.robot_poses.push_back(agent.state.pos);
  Info info;
  info.step_cnt = step_count_;
  step_count_++;
  Done done = false;

  // check if any agent requires a new target
  for (auto &agent : agents_)
  {
    if (agent.action.target_idx == NONE)
    {
      Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
      Alg::map_merge(global_map_, agent.state.map.get());
      auto frontiers = Alg::frontier_detection(agent.state.map.get());
      agent.state.frontier_points = frontiers;
      if (frontiers.empty())
      {
        done = true;
      }
      info.robot_id = agent.info.id;
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
  set_action(agent_id, target_index);
  Observation obs;
  Info info;
  info.step_cnt = step_count_;
  step_count_++;
  Done done;
  Reward reward;
  // check if any agent requires a new target
  for (auto &agent : agents_)
  {
    if (agent.action.target_idx == NONE)
    {
      Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
      auto new_explored_pixels = Alg::calculate_new_explored_pixels(env_map_, agent.state.map.get());
      if (agent.info.step_count != 0)
      {
        agent.reward.explored_pixels += new_explored_pixels;
      }
      reward.exploration_reward = agent.reward.explored_pixels;
      Alg::map_merge(global_map_, agent.state.map.get());
      auto frontiers = Alg::frontier_detection(agent.state.map.get());
      agent.state.frontier_points = frontiers;
      if (frontiers.empty())
      {
        done = true;
      }
      info.robot_id = agent.info.id;
      info.exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
      info.agent_step_cnt = agent.info.step_count;
      agent.info.step_count++;
      return std::make_tuple(obs, reward, done, info);
    }
  }

  // move the agent
  while (true)
  {
    for (auto &agent : agents_)
    {
      if (agent.state.executing_path == nullptr || agent.state.executing_path->empty())
      {
        agent.action.target_idx = NONE;
        break;
      }
      auto next_pos = agent.state.executing_path->back();
      agent.state.executing_path->pop_back();
      if (env_map_->operator()(next_pos.x, next_pos.y) == FREE)
      {
        agent.state.pos = next_pos;
        break;
      }
    }
  }
  return std::make_tuple(obs, reward, done, info);
}
void Environment::test_grid_map()
{
  for (int i = 0; i < env_map_->rows(); i++)
  {
    for (int j = 0; j < env_map_->cols(); j++)
    {
      env_map_->operator()(i, j) = 255;
    }
  }
}

void Environment::set_action(int agent_id, Action target_idx)
{
  agents_[agent_id].action.target_idx = target_idx;
  agents_[agent_id].action.target_x = agents_[agent_id].state.frontier_points[target_idx].pos.x;
  agents_[agent_id].action.target_y = agents_[agent_id].state.frontier_points[target_idx].pos.y;

  agents_[agent_id].state.executing_path =
      std::make_unique<Path>(Alg::a_star(agents_[agent_id].state.map.get(), agents_[agent_id].state.pos,
                                         agents_[agent_id].state.frontier_points[target_idx].pos));
  agents_[agent_id].state.frontier_points.clear();
}

} // namespace Env