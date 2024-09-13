#include "env.hpp"
#include "algorithms.hpp"
#include "spdlog/cfg/env.h"
#include <spdlog/sinks/basic_file_sink.h>

#include <spdlog/spdlog.h>
namespace Env
{

Environment::Environment(int num_agents, int max_steps, int max_steps_per_agent, int velocity, int sensor_range,
                         int num_rays, int min_frontier_pixel, int max_frontier_pixel, float exploration_threshold)
    : num_agents_(num_agents), max_steps_(max_steps), max_steps_per_agent_(max_steps_per_agent), velocity_(velocity),
      sensor_range_(sensor_range), num_rays_(num_rays), min_frontier_pixel_(min_frontier_pixel),
      max_frontier_pixel_(max_frontier_pixel), exploration_threshold_(exploration_threshold)
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
  this->global_map_ = std::make_shared<GridMap>(env_map.width(), env_map.height(), UNKNOWN);
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
  spdlog::info("env_map size: {}x{}", env_map.rows(), env_map.cols());
  spdlog::info("poses:");
  std::string poses_log;
  for (auto &pose : poses)
    poses_log += "(" + std::to_string(pose.x) + ", " + std::to_string(pose.y) + ") ";
  spdlog::info(poses_log);

  this->init(env_map, poses);

  auto agent_id = get_next_act_agent();
  spdlog::info("next act agent: {}", agent_id);

  if (agent_id == -1)
  {
    agent_id = 0;
    spdlog::info("no agent requires a new target, returning frame data for agent 0");
  }
  else
  {
    spdlog::info("agent {} requires a new target", agent_id);
  }

  spdlog::trace("manually updating map for all agents and merge map to global map");
  for (auto &agent : agents_)
  {
    Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
    Alg::map_merge(global_map_, agent.state.map.get());
  }

  return get_frame_data(agent_id);
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
 * @note  this will clear the frontier points and rewards, compute the path and set the target index. We also restore a
 * copy of the agent map for calculating the new explored pixels
 * @param agent_id  the id of the agent
 * @param target_idx  the index of the target frontier point
 */
void Environment::set_action(int agent_id, Action target_idx)
{
  auto &&agent = agents_[agent_id];
  if (agent.done.done)
  {
    spdlog::info("agent {} done, skip setting action.", agent_id);
    return;
  }
  FrontierPoint target_frontier = agent.state.frontier_points[target_idx];
  spdlog::debug("reset agent {} action, state, reward", agent_id);
  agent.action.reset();
  agent.reward.reset();
  agent.state.reset();
  agent.info.reset();

  spdlog::debug("set action for {} action index: {} target pos: {}", agent_id, target_idx, target_frontier.pos);
  agent.action = {target_idx, target_frontier.pos};

  spdlog::info("computing path");
  agent.state.executing_path =
      std::make_unique<Path>(Alg::a_star(agent.state.map.get(), agent.state.pos, target_frontier.pos));
  if (agent.state.executing_path->empty())
  {
    spdlog::info("path is empty");
    agent.info.failed_attempts++;
    if (agent.info.failed_attempts >= MAX_FAILED_ATTEMPTS)
    {
      spdlog::info("agent {} done for max failed attempts received", agent_id);
      spdlog::info("start grid: {}, target grid: {}", agent.state.map->operator()(agent.state.pos.x, agent.state.pos.y),
                   agent.state.map->operator()(target_frontier.pos.x, target_frontier.pos.y));
      agent.done.done = true;
    }
  }
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

  Observation observation;
  Reward reward;
  Done done = false;
  Info info;

  // assume the agent is not done
  auto &&agent = agents_[agent_id];

  info.agent_id = agent.info.id;
  info.step_cnt = step_count_;
  info.delta_time = agent.info.delta_time;
  reward.time_step_reward = info.delta_time;
  step_count_++;
  info.agent_step_cnt = agent.info.step_count;
  agent.info.step_count++;

  spdlog::trace("calculating new explored pixels");
  auto valid_explored_pixels = Alg::calculate_valid_explored_pixels(global_map_, agent.state.map.get());
  agent.reward.new_explored_pixels += valid_explored_pixels;
  reward.exploration_reward = agent.reward.new_explored_pixels;

  agent.info.explored_pixels = Alg::count_pixels(agent.state.map.get(), UNKNOWN, true);
  info.agent_explored_pixels = agent.info.explored_pixels;
  spdlog::trace("all explored pixels: {}, all new explored pixels: {}", info.agent_explored_pixels,
                reward.exploration_reward);

  spdlog::trace("detecting frontiers");
  auto &&frontiers =
      Alg::frontier_detection(agent.state.map.get(), min_frontier_pixel_, max_frontier_pixel_, sensor_range_);
  spdlog::debug("frontiers size: {}", frontiers.size());

  std::vector<FrontierPoint> valid_frontiers;
  for (auto &frontier : frontiers)
  {
    if (Alg::is_frontier_valid(global_map_, frontier, sensor_range_, sensor_range_ * sensor_range_ / 4, false,
                               agent.state.map.get(), agent.state.pos))
    {
      auto path=Alg::a_star(agent.state.map.get(), agent.state.pos, frontier.pos);
      frontier.distance=path.size();
      valid_frontiers.push_back(frontier);
    }
  }

  spdlog::debug("valid frontiers size: {}", valid_frontiers.size());
  agent.state.frontier_points = valid_frontiers;
  observation.frontier_points = valid_frontiers;

  if (valid_frontiers.empty())
  {
    spdlog::info("agent {} done", agent_id);
    agent.done.done = true;
  }

  spdlog::trace("getting agent poses");
  for (auto &agent : agents_)
  {
    observation.agent_poses.push_back(agent.state.pos);
    if (agent.action.target_idx != -1)
      observation.agent_targets.push_back(agent.action.target_pos);
    else
    {
      observation.agent_targets.push_back(Coord(0, 0));
    }
  }
  std::swap(observation.agent_poses[0], observation.agent_poses[agent_id]);
  std::swap(observation.agent_targets[0], observation.agent_targets[agent_id]);

  spdlog::trace("merging map to global map");
  Alg::map_merge(global_map_, agent.state.map.get());

  auto agent_exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
  auto global_exploration_rate = Alg::exploration_rate(env_map_, global_map_.get());
  spdlog::trace("agent exploration rate:{}, global exploration rate: {}", agent_exploration_rate,
                global_exploration_rate);
  info.agent_exploration_rate = agent_exploration_rate;
  info.global_exploration_rate = global_exploration_rate;
  if (global_exploration_rate >= exploration_threshold_)
  {
    spdlog::info("global exploration rate reached threshold");
    agent.done.done = true;
    this->is_done_ = true;
  }
  done = agent.done.done || this->is_done_;

  return std::move(std::make_tuple(observation, reward, done, info));
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
  while (++tick_ < max_steps_)
  {
    spdlog::trace("tick: {}", tick_);
    for (auto &agent : agents_)
    {
      spdlog::trace("agent: {}", agent.info.id);
      if (agent.done.done)
      {
        spdlog::info("agent {} done, skip step.", agent.info.id);
        if (std::find_if(agents_.begin(), agents_.end(), [](const Agent &agent) { return agent.done.done != true; }) ==
                agents_.end() ||
            this->is_done_)
        {
          spdlog::info("all agents done");
          this->is_done_ = true;
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

      auto valid_explored_pixels = Alg::calculate_valid_explored_pixels(global_map_, agent.state.map.get());
      agent.reward.new_explored_pixels += valid_explored_pixels;
      agent.info.explored_pixels = Alg::count_pixels(agent.state.map.get(), UNKNOWN, true);
      agent.info.delta_time++;
      spdlog::info("explored pixels: {} ,new explored pixels: {},all new explored pixels: {}",
                   agent.info.explored_pixels, valid_explored_pixels, agent.reward.new_explored_pixels);

      spdlog::trace("merging map to global map");
      Alg::map_merge(global_map_, agent.state.map.get());
    }
  }
  spdlog::info("step limit reached");
  return -1;
}

} // namespace Env