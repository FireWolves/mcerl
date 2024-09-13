#include "env.hpp"
#include "algorithms.hpp"
#include "spdlog/cfg/env.h"
#include <spdlog/sinks/basic_file_sink.h>

#include <spdlog/spdlog.h>
namespace Env

{
/**
 * @brief Environment class represents the environment in which agents operate.
 *
 * The Environment class provides methods for initializing the environment, resetting it, and stepping through the
 * environment. It also includes methods for setting actions for agents, calculating frame data, and detecting
 * frontiers. The environment is represented by a grid map, and agents can move within the map to explore and reach
 * target frontier points.
 *
 * The Environment class contains the following methods:
 * - Environment(): Constructor for the Environment class.
 * - init(): Initializes the environment with the specified parameters.
 * - reset(): Resets the environment to the initial state.
 * - step(): Performs a single step in the environment for the specified agent.
 * - set_action(): Sets the action for a specific agent.
 * - get_frame_data(): Returns the observation, reward, done flag, and info for a specific agent.
 * - get_next_act_agent(): Returns the ID of the next agent that requires a new target.
 * - step_once(): Performs a single step for all agents in the environment.
 */
Environment::Environment()
{
  // 创建一个文件日志接收器
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_st>("logs/debug.txt", true);

  // 创建一个复合日志器，将日志同时输出到控制台和文件
  spdlog::logger logger("ENV", {file_sink});

  // 替换默认日志器
  spdlog::set_default_logger(std::make_shared<spdlog::logger>(logger));
  spdlog::cfg::load_env_levels();
  spdlog::flush_on(spdlog::level::trace);
  spdlog::trace("Logger initialized");

  spdlog::debug("Environment instanced");
}

void Environment::init(GridMap env_map, std::vector<Coord> poses, int num_agents, int max_steps,
                       int max_steps_per_agent, int velocity, int sensor_range, int num_rays, int min_frontier_pixel,
                       int max_frontier_pixel, float exploration_threshold)

{
  spdlog::debug("Initializing environment");

  spdlog::trace("Setting environment parameters");
  this->num_agents_ = num_agents;
  this->max_steps_ = max_steps;
  this->max_steps_per_agent_ = max_steps_per_agent;
  this->velocity_ = velocity;
  this->sensor_range_ = sensor_range;
  this->num_rays_ = num_rays;
  this->min_frontier_pixel_ = min_frontier_pixel;
  this->max_frontier_pixel_ = max_frontier_pixel;
  this->exploration_threshold_ = exploration_threshold;
  spdlog::trace("Environment parameters set");

  spdlog::trace("Creating env map and global map: {}x{}", env_map.width(), env_map.height());
  this->env_map_ = std::make_shared<GridMap>(env_map);
  this->global_map_ = std::make_shared<GridMap>(env_map.width(), env_map.height(), UNKNOWN);
  spdlog::trace("Map created");

  spdlog::trace("Setting agents' initial poses and resetting track info");
  this->init_poses_ = poses;
  this->reset_state();
  spdlog::trace("Agents' initial poses set and track info reset");

  spdlog::trace("Resetting agents");
  agents_.resize(num_agents);
  for (int i = 0; i < num_agents_; i++)
    agents_[i].reset(this->env_map_, poses[i], i, max_steps_per_agent_, sensor_range_, num_rays_);
  spdlog::trace("Agents reset");

  spdlog::debug("Environment initialized");
}

FrameData Environment::reset(GridMap env_map, std::vector<Coord> poses, int num_agents, int max_steps,
                             int max_steps_per_agent, int velocity, int sensor_range, int num_rays,
                             int min_frontier_pixel, int max_frontier_pixel, float exploration_threshold)
{
  spdlog::debug("Resetting environment");

  spdlog::trace("Initializing environment");
  this->init(env_map, poses, num_agents, max_steps, max_steps_per_agent, velocity, sensor_range, num_rays,
             min_frontier_pixel, max_frontier_pixel, exploration_threshold);
  spdlog::trace("Environment initialized");

  spdlog::trace("Getting next act agent and checking if next act agent is valid");
  auto agent_id = get_next_act_agent();
  if (agent_id == INVALID_AGENT_ID)
  {
    spdlog::debug("Invalid agent ID, set next act agent to 0");
    agent_id = 0;
  }
  spdlog::trace("Next act agent: {}", agent_id);

  spdlog::trace("Updating map for all agents and merge map to global map");
  for (auto &agent : agents_)
  {
    Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);
    Alg::map_merge(global_map_, agent.state.map.get());
  }
  spdlog::trace("Map updated and merged");

  spdlog::trace("Getting frame data for agent {}", agent_id);
  auto &&frame_data = get_frame_data(agent_id);
  spdlog::trace("Frame data retrieved");

  spdlog::debug("Environment reset");
  return frame_data;
}

FrameData Environment::step(int agent_id, Action target_index)
{
  spdlog::debug("Stepping environment for agent {} with target index {}", agent_id, target_index);

  spdlog::trace("Setting agent action");
  set_action(agent_id, target_index);
  spdlog::trace("Agent action set");

  spdlog::trace("Checking if any agent requires a new target");
  auto next_act_agent = get_next_act_agent();
  if (next_act_agent == INVALID_AGENT_ID)
  {
    spdlog::trace("No agent requires a new target, stepping once and getting next act agent");
    next_act_agent = step_once();
    spdlog::trace("Environment stepped once");
  }
  if (next_act_agent == INVALID_AGENT_ID)
  {
    spdlog::debug("No agent requires action, all agents done, set done flag");
    this->is_done_ = true;

    spdlog::trace("Set next act agent to 0 for getting dummy data");
    next_act_agent = 0;
  }

  spdlog::trace("Getting frame data for agent {}", next_act_agent);
  auto &&frame_data = get_frame_data(next_act_agent);
  spdlog::trace("Frame data retrieved");

  spdlog::debug("Environment stepped");
  return frame_data;
}

/**
 * @brief set the action for the agent
 * @note  this will clear the frontier points and rewards, compute the path and set the target index.
 * @param agent_id  the id of the agent
 * @param target_idx  the index of the target frontier point
 */
void Environment::set_action(int agent_id, Action target_idx)
{
  spdlog::debug("Setting action for agent {} with target index {}", agent_id, target_idx);

  auto &&agent = agents_[agent_id];

  if (agent.done.done)
  {
    spdlog::debug("Agent {} done, skip setting action", agent_id);
    return;
  }

  spdlog::trace("Getting target frontier point");
  FrontierPoint target_frontier = agent.state.frontier_points[target_idx];
  spdlog::trace("Target frontier point retrieved, pos: {}, unexplored pixels: {}", target_frontier.pos,
                target_frontier.unexplored_pixels);

  spdlog::trace("Resetting agent {} action, state, reward, info", agent_id);
  agent.action.reset();
  agent.reward.reset();
  agent.state.reset();
  agent.info.reset();
  spdlog::trace("Agent data reset");

  spdlog::trace("Set new action and computing path for {} action index: {} target pos: {}", agent_id, target_idx,
                target_frontier.pos);
  agent.action = {target_idx, target_frontier.pos};
  agent.state.executing_path =
      std::make_unique<Path>(Alg::a_star(agent.state.map.get(), agent.state.pos, target_frontier.pos));
  spdlog::trace("Action and path set, path size: {}", agent.state.executing_path->size());

  spdlog::trace("Checking if path is empty");
  if (agent.state.executing_path->empty())
  {
    agent.info.failed_attempts++;
    spdlog::debug("Path is empty, Failed attempts: {}", agent.info.failed_attempts);

    if (agent.info.failed_attempts >= MAX_FAILED_ATTEMPTS)
    {
      spdlog::warn("Max failed attempts received, setting done flag for agent {}", agent_id);
      spdlog::warn("Start: {}, target: {}", agent.state.map->operator()(agent.state.pos.x, agent.state.pos.y),
                   agent.state.map->operator()(target_frontier.pos.x, target_frontier.pos.y));
      agent.done.done = true;
      spdlog::trace("Done flag set");
    }
  }
  else
  {
    spdlog::trace("Path not empty, resetting failed attempts");
    agent.info.failed_attempts = 0;
  }

  spdlog::debug("Action set for agent {}", agent_id);
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
  spdlog::debug("Getting frame data for agent {}", agent_id);

  Observation observation;
  Reward reward;
  Done done = false;
  Info info;

  auto &&agent = agents_[agent_id];

  info.agent_id = agent.info.id;

  spdlog::trace("Updating delta time and step count");
  info.delta_time = agent.info.delta_time;
  reward.time_step_reward = info.delta_time;

  info.step_cnt = step_count_;
  step_count_++;
  info.agent_step_cnt = agent.info.step_count;
  agent.info.step_count++;
  spdlog::trace("Step count updated: {}, agent step count updated: {}, delta time updated", info.step_cnt,
                info.agent_step_cnt, info.delta_time);

  spdlog::trace("Calculating new explored pixels and recording exploration reward");

  auto valid_explored_pixels = Alg::calculate_valid_explored_pixels(global_map_, agent.state.map.get());
  agent.info.explored_pixels = Alg::count_pixels(agent.state.map.get(), UNKNOWN, true);

  agent.reward.new_explored_pixels += valid_explored_pixels;
  reward.exploration_reward = agent.reward.new_explored_pixels;
  info.agent_explored_pixels = agent.info.explored_pixels;
  spdlog::trace("Valid exploration pixels calculated: {}, new explored pixels: {}, exploration reward: {}, agent all "
                "explored pixels: {}",
                valid_explored_pixels, agent.reward.new_explored_pixels, reward.exploration_reward,
                agent.info.explored_pixels);

  spdlog::trace("Detecting frontiers");
  auto &&frontiers =
      Alg::frontier_detection(agent.state.map.get(), min_frontier_pixel_, max_frontier_pixel_, sensor_range_);
  spdlog::trace("Frontiers detected. size: {}", frontiers.size());

  spdlog::trace("Checking if frontiers are valid");
  std::vector<FrontierPoint> valid_frontiers;
  for (auto &frontier : frontiers)
  {
    if (Alg::is_frontier_valid(global_map_, frontier, sensor_range_, sensor_range_ * sensor_range_ / 4))
    {
      auto path = Alg::a_star(agent.state.map.get(), agent.state.pos, frontier.pos);
      frontier.distance = path.size();
      valid_frontiers.push_back(frontier);
    }
  }
  agent.state.frontier_points = valid_frontiers;
  observation.frontier_points = valid_frontiers;

  if (valid_frontiers.empty())
  {
    spdlog::debug("Frontier empty, setting done flag for agent {}", agent_id);
    agent.done.done = true;
    spdlog::trace("Done flag set");
  }
  spdlog::trace("Frontier checked");

  spdlog::trace("Getting agent poses and agents' targets' poses");
  for (auto &agent : agents_)
  {
    observation.agent_poses.push_back(agent.state.pos);
    if (agent.action.target_idx != INVALID_TARGET)
      observation.agent_targets.push_back(agent.action.target_pos);
    else
    {
      observation.agent_targets.push_back(Coord(0, 0));
    }
  }
  std::swap(observation.agent_poses[0], observation.agent_poses[agent_id]);
  std::swap(observation.agent_targets[0], observation.agent_targets[agent_id]);
  spdlog::trace("Agent poses and targets retrieved");

  spdlog::trace("Merging map to global map");
  Alg::map_merge(global_map_, agent.state.map.get());
  spdlog::trace("Map merged");

  spdlog::trace("Calculating exploration rate for agent and global map and checking if threshold reached");
  auto agent_exploration_rate = Alg::exploration_rate(env_map_, agent.state.map.get());
  auto global_exploration_rate = Alg::exploration_rate(env_map_, global_map_.get());
  info.agent_exploration_rate = agent_exploration_rate;
  info.global_exploration_rate = global_exploration_rate;
  spdlog::trace("Exploration rate calculated. agent exploration rate:{}, global exploration rate: {}",
                agent_exploration_rate, global_exploration_rate);

  if (global_exploration_rate >= exploration_threshold_)
  {
    spdlog::debug("Global exploration rate reached threshold, setting done flag for agent {} and environment ",
                  agent_id);
    agent.done.done = true;
    this->is_done_ = true;
    spdlog::trace("Done flag set");
  }
  done = agent.done.done || this->is_done_;
  spdlog::trace("Exploration rate checked");

  spdlog::debug("Frame data retrieved for agent {}", agent_id);
  return std::move(std::make_tuple(observation, reward, done, info));
}
int Environment::get_next_act_agent()
{
  spdlog::debug("Getting next act agent");

  spdlog::trace("Finding agent that requires a new target");
  int next_act_agent_id = INVALID_AGENT_ID;
  auto res = std::find_if(agents_.begin(), agents_.end(),
                          [](const Agent &x) { return x.action.target_idx == INVALID_TARGET && x.done.done != true; });
  if (res != agents_.end())
  {
    next_act_agent_id = res->info.id;
  }
  spdlog::debug("Next act agent found: {}", next_act_agent_id);

  return next_act_agent_id;
}

int Environment::step_once()
{
  spdlog::debug("Step env once");

  spdlog::trace("Ticking env");
  while (++tick_ < max_steps_)
  {
    spdlog::trace("Checking if all agents are done");
    if (this->is_done_ || std::find_if(agents_.begin(), agents_.end(),
                                       [](const Agent &agent) { return agent.done.done != true; }) == agents_.end())
    {
      spdlog::debug("All agents done or environment done, setting done flag for environment and exit step once");
      this->is_done_ = true;
      return INVALID_AGENT_ID;
    }
    spdlog::trace("Agents done checked. Looping agents at tick: {}", tick_);

    for (auto &agent : agents_)
    {
      spdlog::trace("Agent: {}", agent.info.id);

      spdlog::trace("Checking if agent reached target");
      if (agent.state.executing_path == nullptr || agent.state.executing_path->empty())
      {
        spdlog::debug("Agent {} reached target. Exit step once", agent.info.id);
        return agent.info.id;
      }

      spdlog::trace("Moving agent");
      auto next_pos = agent.state.executing_path->front();
      agent.state.executing_path->erase(agent.state.executing_path->begin());
      agent.state.pos = next_pos;

      spdlog::trace("Updating map");
      Alg::map_update(env_map_, agent.state.map.get(), agent.state.pos, agent.info.sensor_range, agent.info.num_rays);

      spdlog::trace("Calculating valid explored pixels and recording delta time");
      auto valid_explored_pixels = Alg::calculate_valid_explored_pixels(global_map_, agent.state.map.get());
      agent.reward.new_explored_pixels += valid_explored_pixels;
      agent.info.delta_time++;
      spdlog::info("Valid explored pixels: {}, all new explored pixels: {}", valid_explored_pixels,
                   agent.reward.new_explored_pixels);

      spdlog::trace("Merging map to global map");
      Alg::map_merge(global_map_, agent.state.map.get());
      spdlog::trace("Map merged");
    }
  }
  spdlog::warn("Step limit reached, setting done flag for environment and exit step once");
  this->is_done_ = true;
  return INVALID_AGENT_ID;
}

} // namespace Env