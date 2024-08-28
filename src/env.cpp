#include "env.hpp"
#include "algorithms.hpp"
#include "spdlog/cfg/env.h"
#include <spdlog/sinks/basic_file_sink.h>

#include <spdlog/spdlog.h>
namespace Env
{

Environment::Environment(int num_agents, int max_steps, int max_steps_per_agent, int velocity, int sensor_range,
                         int num_rays, int min_frontier_pixel, int max_frontier_pixel)
    : num_agents_(num_agents), max_steps_(max_steps), max_steps_per_agent_(max_steps_per_agent), velocity_(velocity),
      sensor_range_(sensor_range), num_rays_(num_rays), min_frontier_pixel_(min_frontier_pixel),
      max_frontier_pixel_(max_frontier_pixel)
{
  // 创建一个控制台日志接收器
  // auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  // 创建一个文件日志接收器
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_st>("logs/debug.txt", true);

  // 创建一个复合日志器，将日志同时输出到控制台和文件
  spdlog::logger logger("env", {file_sink});

  // 设置日志格式（可选）
  // logger.set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  // 替换默认日志器
  spdlog::set_default_logger(std::make_shared<spdlog::logger>(logger));
  spdlog::cfg::load_env_levels();
  spdlog::flush_on(spdlog::level::trace);
  agents_.resize(num_agents);
}

void Environment::init(GridMap env_map, std::vector<Coord> poses)
{
  spdlog::debug("init");

  this->env_map_ = std::make_shared<GridMap>(env_map);
  this->global_map_ = std::make_shared<GridMap>(env_map.rows(), env_map.cols(), UNKNOWN);
  spdlog::trace("map created");

  this->init_poses_ = poses;
  spdlog::trace("posed setted");

  spdlog::debug("resetting track info");
  this->reset_state();

  spdlog::debug("resetting agents");
  for (int i = 0; i < num_agents_; i++)
    agents_[i].reset(this->env_map_, poses[i], i, max_steps_per_agent_, sensor_range_, num_rays_);
}

FrameData Environment::reset(GridMap env_map, std::vector<Coord> poses)
{
  spdlog::info("reset");

  this->init(env_map, poses);

  auto next_act_agent = get_next_act_agent();
  spdlog::info("next act agent: {}", next_act_agent);

  if (next_act_agent == -1)
  {
    next_act_agent = 0;
    spdlog::info("no agent requires a new target, returning frame data for agent 0");
  }
  else
  {
    spdlog::info("agent {} requires a new target", next_act_agent);
  }

  return get_frame_data(next_act_agent);
}

FrameData Environment::step(int agent_id, Action target_index)
{
  spdlog::info("step");

  spdlog::info("set action agent_id: {} target_index: {}", agent_id, target_index);
  set_action(agent_id, target_index);

  spdlog::info("check if agent requires a new target");
  auto next_act_agent = get_next_act_agent();
  if (next_act_agent != -1)
  {
    spdlog::debug("agent {} requires a new target", next_act_agent);
    return get_frame_data(next_act_agent);
  }
  // move the agent
  spdlog::info("moving agent");
  next_act_agent = step_once();
  if (next_act_agent == -1)
  {
    spdlog::info("all agents done");
    this->is_done_ = true;
    return get_frame_data(0);
  }
  return get_frame_data(next_act_agent);
}

/**
 * @brief set the action for the agent
 * @note  this will clear the frontier points and rewards, compute the path and set the target index
 * @param agent_id  the id of the agent
 * @param target_idx  the index of the target frontier point
 */
void Environment::set_action(int agent_id, Action target_idx)
{
  auto &&agent = agents_[agent_id];
  FrontierPoint target_frontier = agent.state.frontier_points[target_idx];
  spdlog::debug("reset agent {} action, state, reward", agent_id);
  agent.action.reset();
  agent.reward.reset();
  agent.state.reset();
  agent.info.reset();

  spdlog::debug("set_action for {} action index: {} target pos: {}", agent_id, target_idx, target_frontier.pos);
  agent.action = {target_idx, target_frontier.pos};

  spdlog::info("computing path");
  agent.state.executing_path =
      std::make_unique<Path>(Alg::a_star(agent.state.map.get(), agent.state.pos, target_frontier.pos));
  spdlog::debug("path size: {}", agent.state.executing_path->size());
}

/**
 * @brief return the observation, reward, done flag and info for specific agent
 * @param agent_id the id of the agent
 * @note this will update the agent map, merge agent map to global map, calculate the whole exploration rate, detect
 * frontiers and check if the agent is done
 * @note the agent_id should be valid and not done
 * @return FrameData
 */
FrameData Environment::get_frame_data(int agent_id)
{
  spdlog::info("get frame data for agent {}", agent_id);

  Observation obs;
  Reward reward;
  Done done = false;
  Info info;

  // assume the agent is not done
  auto &&agent = agents_[agent_id];

  info.robot_id = agent.info.id;
  info.step_cnt = step_count_;
  info.delta_time = agent.info.delta_time;
  step_count_++;
  info.agent_step_cnt = agent.info.step_count;
  agent.info.step_count++;

  // manually update the map
  spdlog::trace("updating map");
  Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);

  spdlog::trace("calculating new explored pixels");
  auto new_explored_pixels = Alg::calculate_new_explored_pixels(global_map_, agent.state.map.get());
  agent.reward.explored_pixels += new_explored_pixels;
  reward.exploration_reward = agent.reward.explored_pixels;

  spdlog::trace("detecting frontiers");
  auto &&frontiers =
      Alg::frontier_detection(agent.state.map.get(), min_frontier_pixel_, max_frontier_pixel_, sensor_range_);
  spdlog::debug("frontiers size: {}", frontiers.size());

  std::vector<FrontierPoint> valid_frontiers;
  for (auto &frontier : frontiers)
    if (Alg::is_frontier_valid(global_map_, frontier, sensor_range_, sensor_range_ * sensor_range_ / 4))
      valid_frontiers.push_back(frontier);
  spdlog::debug("valid frontiers size: {}", valid_frontiers.size());
  agent.state.frontier_points = valid_frontiers;
  obs.frontier_points = valid_frontiers;

  if (valid_frontiers.empty())
  {
    spdlog::info("agent {} done", agent_id);
    agent.done.done = true;
    done = true;
  }

  spdlog::trace("getting agent poses");
  for (auto &i : agents_)
  {
    obs.robot_poses.push_back(i.state.pos);
  }
  std::swap(obs.robot_poses[0], obs.robot_poses[agent_id]);

  spdlog::trace("merging map to global map");
  Alg::map_merge(global_map_, agent.state.map.get());

  spdlog::trace("calculating exploration rate");
  auto exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
  info.exploration_rate = exploration_rate;

  return std::move(std::make_tuple(obs, reward, done, info));
}
int Environment::get_next_act_agent()
{
  auto res = std::find_if(agents_.begin(), agents_.end(),
                          [](const Agent &x) { return x.action.target_idx == -1 && x.done.done != true; });
  if (res != agents_.end())
    return res->info.id;
  return -1;
}

int Environment::step_once()
{
  spdlog::trace("step once");
  while (++tick_ < INT_MAX)
  {
    spdlog::trace("tick: {}", tick_);
    for (auto &agent : agents_)
    {
      spdlog::trace("agent: {}", agent.info.id);
      if (agent.done.done)
      {
        spdlog::info("agent {} done", agent.info.id);
        if (std::find_if(agents_.begin(), agents_.end(), [](const Agent &agent) { return agent.done.done != true; }) ==
            agents_.end())
        {
          spdlog::info("all agents done");
          return -1;
        }
        continue;
      }
      if (agent.state.executing_path == nullptr || agent.state.executing_path->empty())
      {
        spdlog::info("agent {} reached target", agent.info.id);
        return agent.info.id;
      }

      spdlog::info("moving agent");

      auto next_pos = agent.state.executing_path->front();
      agent.state.executing_path->erase(agent.state.executing_path->begin());
      agent.state.pos = next_pos;

      spdlog::trace("updating map");
      Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);

      auto new_explored_pixels = Alg::calculate_new_explored_pixels(global_map_, agent.state.map.get());
      spdlog::info("new explored pixels: {}", new_explored_pixels);
      agent.reward.explored_pixels += new_explored_pixels;
      agent.info.delta_time++;

      spdlog::trace("merging map to global map");
      Alg::map_merge(global_map_, agent.state.map.get());
    }
  }
  spdlog::info("step limit reached");
  return -1;
}

} // namespace Env