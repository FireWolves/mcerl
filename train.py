# %%
# imports
from __future__ import annotations

import concurrent.futures
import datetime
import logging
import pathlib as pl
import random
import string
import time
from math import pi
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from rich.console import Console
from rich.table import Table
from rl.actor_critic import GAE, Actor, ActorCritic, Critic
from rl.network import GINPolicyNetwork, GINValueNetwork
from rl.utils import Sampler, to_graph
from torch import Tensor
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from mcerl.env import Env
from mcerl.utils import (
    delta_time_reward_standardize,
    exploration_reward_rescale,
    reward_sum,
)

# %%
# load map
maps = []
easy_maps = [
    "map/easy/12.png",
    "map/easy/95.png",
    "map/easy/83.png",
    "map/easy/69.png",
    "map/easy/50.png",
]
hard_maps = ["map/hard/53.png", "map/40.png"]
medium_maps = ["map/medium/12.png", "map/40.png", "map/medium/11.png"]

map_height, map_width = 200, 300

for map_img in easy_maps + hard_maps + medium_maps:
    img = Image.open(map_img)
    img = ImageOps.grayscale(img)
    img = img.resize((map_width, map_height))
    grid_map = np.array(img)
    grid_map[grid_map < 100] = 0
    grid_map[grid_map >= 100] = 255
    maps.append(grid_map)

# %%
# define parameters

#########################
# log参数
#########################

# workspace
workspace_dir = pl.Path.cwd()
# output
output_dir = pl.Path(workspace_dir) / "output"
current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
random_string = "".join(random.choices(string.ascii_letters + string.digits, k=6))
session_name = f"{current_date}_{random_string}"
# experiment
experiment_dir = output_dir / session_name
log_dir = experiment_dir / "log"
cpp_env_log_path = log_dir / "env.log"
py_env_log_path = log_dir / "env_py.log"
model_dir = experiment_dir / "model"
output_images_dir = experiment_dir / "images"
tensorboard_dir = experiment_dir / "tensorboard"

experiment_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
output_images_dir.mkdir(parents=True, exist_ok=True)
tensorboard_dir.mkdir(parents=True, exist_ok=True)


# logger
log_level = "warning"
cpp_env_log_level = "warning"
py_env_log_level = log_level
logger = logging.getLogger()
logger.setLevel(py_env_log_level.upper())
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.addHandler(logging.FileHandler(py_env_log_path, mode="w"))

#########################
# 环境参数
#########################


exclude_parameters = list(locals().keys())

# 几个agent
num_agents = 3

# agent的初始位置
agent_poses = None

# agent的初始方向
num_rays = 16

# 一个env最多迭代多少步
max_steps = 100000

# 一个agent最多迭代多少步
max_steps_per_agent = 100

# 传感器的范围
ray_range = 30

# 速度(pixel per step)
velocity = 1

# 最小的frontier pixel数
min_frontier_size = 8

# 最大的frontier pixel数
max_frontier_size = 30

# 探索的阈值
exploration_threshold = 0.95


# 一个frontier最多可以获得多少信息增益
max_exploration_gain = ray_range**2 * pi


#########################
# PPO参数
#########################

# gae权重
lmbda = 0.98

# discount factor
gamma = 0.96

# reward scale
# 0.5*tanh((x-average_val)/stddev_val)+0.5
average_val = 852
stddev_val = 765
# 1/(1+((x-x_star)/sigma)^(2*b))
b = 1
x_star = 10
sigma = 40

# clip范围
clip_coefficient = 0.2

# 最大gradient范数
max_grad_norm = 1.0

# ESS
entropy_coefficient = 0.03

entropy_coefficient_decay = 0.995

# policy loss的权重
policy_loss_weight = 1.0

# value loss的权重
value_loss_weight = 1.0

# learning rate
lr = 5e-4
#########################
# 训练参数
#########################

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 总共训练的次数
n_iters = 1000
# n_iters = 10


# 每次训练的并行环境数(num worker)
n_parallel_envs = 96

# 每次训练的epoch数, 即数据要被训练多少次
n_epochs_per_iter = 8

# 每次epoch的mini_batch大小
n_frames_per_mini_batch = 5120

# 每个agent的最大步数
max_steps_per_agent = 40

# 每次训练所用到的总的环境数量
# n_envs = round(n_frames_per_iter / num_agents / max_steps_per_agent)

n_envs = 512
n_eval_envs = 512
# 每次训练的frame数(大约)
n_frames_per_iter = max_steps_per_agent * num_agents * n_envs
# 每个epoch的frame数
n_frames_per_epoch = n_frames_per_iter

# 每个epoch的mini_batch数
n_mini_batches_per_epoch = n_frames_per_epoch // n_frames_per_mini_batch


parameters = {k: v for k, v in locals().items() if k not in [*exclude_parameters, "exclude_parameters"]}
console = Console()
console.print(parameters)

# %%
# transform function


def expert_policy(frame_data: dict[str, Any]) -> dict[str, Any]:
    """
    greedy policy
    utility : u=(A-beta*D)*(0.85 if overlap else 1)
    overlap: if the frontier point is close to the robot or close to robots' target
    """
    beta = 10
    sensor_range = 30
    discount_factor = 0.85
    utilities = []
    if len(frame_data["observation"]["frontier_points"]) == 0:
        frame_data["expert_action"] = 0
        return frame_data

    for frontier in frame_data["observation"]["frontier_points"]:
        pos = frontier[:2]
        area = frontier[2]
        distance = frontier[3]
        utility = area - beta * distance
        for pose in frame_data["observation"]["pos"][1:] + frame_data["observation"]["target_pos"][1:]:
            distance = np.linalg.norm(np.array(pos) - np.array(pose))
            if distance < sensor_range:
                utility *= discount_factor
                break
        utilities.append(utility)
    action_index = np.argmax(utilities)
    # utilities = torch.tensor(utilities).float()
    # with torch.no_grad():
    #     probabilities = torch.nn.functional.softmax(utilities, dim=0).reshape(-1)
    #     action_index = torch.argmax(probabilities)

    frame_data["expert_action"] = action_index
    return frame_data


# %%
def env_transform(frame_data: dict[str, Any]) -> dict[str, Any]:
    """
    normalize position, exploration gain,etc.
    Note that we need to check if env is done
    """
    frame_data = expert_policy(frame_data)

    width = float(map_width)
    height = float(map_height)

    # normalize frontier position to [0,1] and exploration gain to [0,1]
    frame_data["observation"]["frontier_points"] = [
        (
            float(x) / width,
            float(y) / height,
            float(gain) / max_exploration_gain,
            float(distance) / height,
        )
        for x, y, gain, distance in frame_data["observation"]["frontier_points"]
    ]

    # normalize position to [0,1]
    frame_data["observation"]["pos"] = [
        (float(x) / width, float(y) / height) for x, y in frame_data["observation"]["pos"]
    ]

    frame_data["observation"]["target_pos"] = [
        (float(x) / width, float(y) / height) for x, y in frame_data["observation"]["target_pos"]
    ]

    # build graph
    frame_data = to_graph(frame_data)

    return frame_data  # noqa: RET504 explicit return


# policy transform
def device_cast(frame_data: dict[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    """
    cast data to device
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if "graph" in frame_data["observation"]:
        frame_data["observation"]["graph"] = frame_data["observation"]["graph"].to(device)

    return frame_data


# %%
# define policy

policy_network = GINPolicyNetwork(dim_feature=6, dim_h=72)
value_network = GINValueNetwork(dim_feature=6, dim_h=72)
actor = Actor(policy_network=policy_network)
critic = Critic(value_network=value_network)
wrapped_actor_critic = ActorCritic(actor=actor, critic=critic, forward_preprocess=device_cast)
wrapped_actor_critic = wrapped_actor_critic.to(device)
value_estimator = GAE(gamma=gamma, lmbda=lmbda)
optimizer = Adam(wrapped_actor_critic.parameters(), lr=lr)

# %%
# load state dict
# wrapped_actor_critic.load_state_dict(torch.load("ppo_parallel_100.pt"))

# %%
# parameters for env rollout
rollout_parameters = {
    "policy": wrapped_actor_critic,
    "env_transform": env_transform,
    "agent_poses": agent_poses,
    "num_agents": num_agents,
    "max_steps": max_steps,
    "max_steps_per_agent": max_steps_per_agent,
    "velocity": velocity,
    "sensor_range": ray_range,
    "num_rays": num_rays,
    "min_frontier_pixel": min_frontier_size,
    "max_frontier_pixel": max_frontier_size,
    "exploration_threshold": exploration_threshold,
    "return_maps": False,
    "requires_grad": False,
    "action_key": "expert_action",
}
console.print(rollout_parameters)

# %%
env = Env(
    cpp_env_log_level,
    cpp_env_log_path.as_posix(),
)
rollout_parameters["grid_map"] = random.choice(maps)

# %%
# plt.imshow(rollout_parameters["grid_map"])

# %%
# single rollout
# with torch.no_grad():
#     single_rollouts = env.rollout(**rollout_parameters)
# rollouts = single_rollouts

# %%
# parallel rollout


def rollout_env(env, params):
    """在给定的环境中执行rollout。"""
    random_index = random.choice(list(range(len(maps))))
    # console.print(f"随机选择地图索引: {random_index}")
    random_map = maps[random_index]

    params["grid_map"] = random_map
    return env.rollout(**params)


def parallel_rollout(envs, num_parallel_envs, rollout_params):
    """
    在多个仿真环境中并行执行rollout。

    :param envs: List[object] - 仿真环境列表, 每个环境对象应有一个rollout方法。
    :param num_parallel_envs: int - 最大并行环境数量。
    :param rollout_params: dict - 传递给每个环境rollout方法的参数。
    :return: List - 每个环境的rollout结果列表。
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_envs) as executor:
        # 提交所有环境的rollout任务
        futures = {executor.submit(rollout_env, env, rollout_params): env for env in envs}

        # 收集所有任务的结果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                print(f"环境 {futures[future]} 的rollout执行时发生错误: {e}")  # noqa: T201

    return results


envs = [Env(cpp_env_log_level, cpp_env_log_path=cpp_env_log_path.as_posix()) for i in range(n_envs)]
len(envs)

# %%
# parallel rollout
# with torch.no_grad():
#     rollouts=parallel_rollout(
#         envs=envs,
#         num_parallel_envs=n_parallel_envs,
#         rollout_params=rollout_parameters,
# )
# len(rollouts)

# %%
# for logging
writer = SummaryWriter(tensorboard_dir / "ppo")
writer.add_text("parameters", str(parameters))
logging_data = {}


# %%
def compute_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    entropies: Tensor,
    returns: Tensor,
    new_values: Tensor,
):
    """
    global_step
    policy_loss_weight
    value_loss_weight
    ess_weight
    """
    loss_compute_start_time = time.time()
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()

    with torch.no_grad():
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = [((ratio - 1.0).abs() > clip_coefficient).float().mean().item()]

    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coefficient, 1 + clip_coefficient)
    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

    ess_loss = -entropies.mean()

    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

    loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss + ess_weight * ess_loss
    loss_compute_time = time.time() - loss_compute_start_time

    logging_data.update(
        {
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": np.mean(clip_fraction),
            "loss": loss.item(),
            "loss_compute_time": loss_compute_time,
        }
    )
    global global_step  # noqa: PLW0602
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/loss", loss.item(), global_step)
    writer.add_scalar("losses/clip_fraction", np.mean(clip_fraction).item(), global_step)

    return loss


# %%
def train_ppo_minibatch(mini_batch_data: dict[str, Any]):
    mini_batch_start_time = time.time()

    # prepare data
    graphs = mini_batch_data["graphs"].to(device)
    advantages = mini_batch_data["advantages"].to(device).flatten()
    prev_log_prob = mini_batch_data["log_probs"].to(device).flatten()
    frame_indices = mini_batch_data["frame_indices"].to(device)
    returns = mini_batch_data["returns"].to(device).flatten()

    # parallel forward
    forward_start_time = time.time()

    new_action, new_log_probs, new_values, entropies = wrapped_actor_critic.forward_parallel(graphs, frame_indices)

    new_log_probs = new_log_probs.to(device).flatten()
    new_values = new_values.to(device).flatten()
    entropies = entropies.to(device).flatten()
    forward_time = time.time() - forward_start_time

    # compute loss
    loss = compute_loss(
        new_log_probs,
        prev_log_prob,
        advantages,
        entropies,
        returns,
        new_values,
    )

    # optimizer update
    optimizer_start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapped_actor_critic.parameters(), max_grad_norm)
    optimizer.step()
    optimizer_time = time.time() - optimizer_start_time

    mini_batch_time = time.time() - mini_batch_start_time

    logging_data.update(
        {
            "optimizer_time": optimizer_time,
            "forward_time": forward_time,
            "mini_batch_time": mini_batch_time,
        }
    )


# %%
def print_data(logging_data):
    table = Table(title=f"PPO Training on {device}: {n_envs} envs in {n_parallel_envs} threads")
    table.add_column("Key", justify="left")
    table.add_column("Value", justify="left")
    for k, v in logging_data.items():
        table.add_row(k, str(v)[:16])
    console.print(table)


# %%
def train_bc_minibatch(mini_batch_data: dict[str, Any]):
    mini_batch_start_time = time.time()

    # prepare data
    graphs = mini_batch_data["graphs"].to(device)
    frame_indices = mini_batch_data["frame_indices"].to(device)
    returns = mini_batch_data["returns"].to(device).flatten()
    expert_actions = mini_batch_data["expert_actions"].to(device).flatten()

    # parallel forward
    forward_start_time = time.time()

    new_action, new_log_probs, new_values, entropies = wrapped_actor_critic.forward_parallel(
        graphs, frame_indices, expert_actions=expert_actions
    )

    new_log_probs = new_log_probs.to(device).flatten()
    new_values = new_values.to(device).flatten()
    entropies = entropies.to(device).flatten()
    forward_time = time.time() - forward_start_time
    bc_loss = torch.mean(-new_log_probs)
    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

    loss = bc_loss + value_loss_weight * value_loss

    # optimizer update
    optimizer_start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapped_actor_critic.parameters(), max_grad_norm)
    optimizer.step()
    optimizer_time = time.time() - optimizer_start_time

    mini_batch_time = time.time() - mini_batch_start_time

    logging_data.update(
        {
            "optimizer_time": optimizer_time,
            "forward_time": forward_time,
            "mini_batch_time": mini_batch_time,
            "bc_loss": bc_loss.item(),
            "value_loss": value_loss.item(),
            "loss": loss.item(),
        }
    )


# %%
def train_epoch(sampler, n_mini_batches_per_epoch):
    epoch_start_time = time.time()
    # sample data
    sample_start_time = time.time()
    mini_batch_data = sampler.random_sample()
    sample_time = time.time() - sample_start_time

    # train minibatch
    for _n_mini_batch in range(n_mini_batches_per_epoch):
        if bc_finished_flag:
            train_ppo_minibatch(mini_batch_data)
        else:
            train_bc_minibatch(mini_batch_data)

    epoch_end_time = time.time() - epoch_start_time

    logging_data.update(
        {
            "sample_time": sample_time,
            "epoch_time": epoch_end_time,
        }
    )


# %%
def data_post_process(rollouts):
    data_post_process_start_time = time.time()
    with torch.no_grad():
        # normalize rewards
        rollouts = [exploration_reward_rescale(rollout, a=average_val, b=stddev_val) for rollout in rollouts]
        rollouts = [delta_time_reward_standardize(rollout, b=b, sigma=sigma, x_star=x_star) for rollout in rollouts]
        rollouts = [reward_sum(rollout, gamma=gamma) for rollout in rollouts]

        # compute GAE
        gae_start_time = time.time()
        rollouts = [value_estimator(rollout) for rollout in rollouts]
        gae_time = time.time() - gae_start_time

        # flatten rollouts
        flattened_rollouts = [frame_data for rollout in rollouts for frame_data in rollout]

        # compute rewards
        exploration_rewards = np.mean(
            [frame_data["next"]["reward"]["exploration_reward"] for frame_data in flattened_rollouts]
        )
        time_step_rewards = np.mean(
            [frame_data["next"]["reward"]["time_step_reward"] for frame_data in flattened_rollouts]
        )
        total_rewards = np.mean([frame_data["next"]["reward"]["total_reward"] for frame_data in flattened_rollouts])

        # convert to tensor
        graphs = []
        rewards = []
        values = []
        advantages = []
        log_probs = []
        returns = []
        total_times = []
        expert_actions = []
        for frame_data in flattened_rollouts:
            graphs.append(frame_data["observation"]["graph"])
            rewards.append(frame_data["next"]["reward"]["total_reward"])
            values.append(frame_data["value"])
            advantages.append(frame_data["advantage"])
            log_probs.append(frame_data["log_prob"])
            returns.append(frame_data["return"])
            total_times.append(frame_data["info"]["total_time_step"])
            expert_actions.append(frame_data["expert_action"])
        rewards = torch.tensor(rewards).to(device)
        values = torch.tensor(values).to(device)
        advantages = torch.tensor(advantages).to(device)
        log_probs = torch.tensor(log_probs).to(device)
        returns = torch.tensor(returns).to(device)
        total_times = torch.tensor(total_times).to(device)
        expert_actions = torch.tensor(expert_actions).to(device)

        # add to sampler
        sampler = Sampler(
            batch_size=n_frames_per_mini_batch,
            # batch_size=20,  # [DEBUG]
            length=len(flattened_rollouts),
            graphs=graphs,
            rewards=rewards,
            values=values,
            advantages=advantages,
            log_probs=log_probs,
            returns=returns,
            expert_actions=expert_actions,
            total_times=total_times,
        )
    data_post_process_time = time.time() - data_post_process_start_time
    episode_reward_mean = torch.mean(returns).item()
    average_exploration_time = torch.mean(total_times.to(torch.float)).item()

    logging_data.update(
        {
            "exploration_rewards": exploration_rewards.item(),
            "time_step_rewards": time_step_rewards.item(),
            "total_rewards": total_rewards.item(),
            "data_post_process_time": data_post_process_time,
            "gae_time": gae_time,
            "average_exploration_time": average_exploration_time,
            "episode_reward_mean": episode_reward_mean,
        }
    )
    writer.add_scalar("rewards/exploration_rewards", exploration_rewards, global_step)
    writer.add_scalar("rewards/time_step_rewards", time_step_rewards, global_step)
    writer.add_scalar("rewards/total_rewards", total_rewards, global_step)
    writer.add_scalar("rewards/average_exploration_time", average_exploration_time, global_step)
    writer.add_scalar("rewards/episode_reward_mean", episode_reward_mean, global_step)

    return sampler


# %%
def train(rollouts):
    with torch.no_grad():
        sampler = data_post_process(rollouts)
        actual_frames_per_iter = sampler._length
        n_mini_batches_per_epoch = actual_frames_per_iter // n_frames_per_mini_batch
        # n_mini_batches_per_epoch = 1  # [DEBUG]

    for _n_epoch in range(n_epochs_per_iter):
        train_epoch(sampler, n_mini_batches_per_epoch)
    logging_data.update(
        {
            "actual_frames_per_iter": sampler._length,
            "n_mini_batches_per_epoch": n_mini_batches_per_epoch,
        }
    )
    print_data(logging_data)


# %%
def evaluate(n_eval_envs):
    rollout_parameters["action_key"] = "action"

    # neural network policy
    with torch.no_grad():
        rollouts = parallel_rollout(
            envs=envs[:n_eval_envs],
            num_parallel_envs=n_parallel_envs,
            rollout_params=rollout_parameters,
        )
        # normalize rewards
        rollouts = [exploration_reward_rescale(rollout, a=average_val, b=stddev_val) for rollout in rollouts]
        rollouts = [delta_time_reward_standardize(rollout, b=b, sigma=sigma, x_star=x_star) for rollout in rollouts]
        rollouts = [reward_sum(rollout, gamma=gamma) for rollout in rollouts]

        # compute GAE
        rollouts = [value_estimator(rollout) for rollout in rollouts]

        # flatten rollouts
        flattened_rollouts = [frame_data for rollout in rollouts for frame_data in rollout]

        # compute rewards
        nn_exploration_rewards = np.mean(
            [frame_data["next"]["reward"]["exploration_reward"] for frame_data in flattened_rollouts]
        )
        nn_time_step_rewards = np.mean(
            [frame_data["next"]["reward"]["time_step_reward"] for frame_data in flattened_rollouts]
        )
        nn_total_rewards = np.mean([frame_data["next"]["reward"]["total_reward"] for frame_data in flattened_rollouts])
        nn_total_times = np.mean([frame_data["info"]["total_time_step"] for frame_data in flattened_rollouts])
        nn_returns = np.mean([frame_data["return"] for frame_data in flattened_rollouts])

    # expert policy
    rollout_parameters["action_key"] = "expert_action"
    with torch.no_grad():
        rollouts = parallel_rollout(
            envs=envs[:n_eval_envs],
            num_parallel_envs=n_parallel_envs,
            rollout_params=rollout_parameters,
        )
        # normalize rewards
        rollouts = [exploration_reward_rescale(rollout, a=average_val, b=stddev_val) for rollout in rollouts]
        rollouts = [delta_time_reward_standardize(rollout, b=b, sigma=sigma, x_star=x_star) for rollout in rollouts]
        rollouts = [reward_sum(rollout, gamma=gamma) for rollout in rollouts]

        # compute GAE
        rollouts = [value_estimator(rollout) for rollout in rollouts]

        # flatten rollouts
        flattened_rollouts = [frame_data for rollout in rollouts for frame_data in rollout]

        # compute rewards
        expert_exploration_rewards = np.mean(
            [frame_data["next"]["reward"]["exploration_reward"] for frame_data in flattened_rollouts]
        )
        expert_time_step_rewards = np.mean(
            [frame_data["next"]["reward"]["time_step_reward"] for frame_data in flattened_rollouts]
        )
        expert_total_rewards = np.mean(
            [frame_data["next"]["reward"]["total_reward"] for frame_data in flattened_rollouts]
        )
        expert_total_times = np.mean([frame_data["info"]["total_time_step"] for frame_data in flattened_rollouts])
        expert_returns = np.mean([frame_data["return"] for frame_data in flattened_rollouts])

    bc_finished_flag = nn_total_times < expert_total_times * 0.8

    global logging_data  # noqa: PLW0603

    logging_data = {}

    logging_data.update(
        {
            "mode": "eval",
            "nn_exploration_rewards": nn_exploration_rewards,
            "nn_time_step_rewards": nn_time_step_rewards,
            "nn_total_rewards": nn_total_rewards,
            "nn_total_times": nn_total_times,
            "nn_returns": nn_returns,
            "expert_exploration_rewards": expert_exploration_rewards,
            "expert_time_step_rewards": expert_time_step_rewards,
            "expert_total_rewards": expert_total_rewards,
            "expert_total_times": expert_total_times,
            "expert_returns": expert_returns,
            "bc_finished": bc_finished_flag,
        }
    )
    writer.add_scalar("eval/nn_exploration_rewards", nn_exploration_rewards, global_step)
    writer.add_scalar("eval/nn_time_step_rewards", nn_time_step_rewards, global_step)
    writer.add_scalar("eval/nn_total_rewards", nn_total_rewards, global_step)
    writer.add_scalar("eval/nn_total_times", nn_total_times, global_step)
    writer.add_scalar("eval/nn_returns", nn_returns, global_step)
    writer.add_scalar("eval/expert_exploration_rewards", expert_exploration_rewards, global_step)
    writer.add_scalar("eval/expert_time_step_rewards", expert_time_step_rewards, global_step)
    writer.add_scalar("eval/expert_total_rewards", expert_total_rewards, global_step)
    writer.add_scalar("eval/expert_total_times", expert_total_times, global_step)
    writer.add_scalar("eval/expert_returns", expert_returns, global_step)
    writer.add_scalar("eval/bc_finished", bc_finished_flag, global_step)

    return bc_finished_flag


# %%
start_time = time.time()
bc_finished_flag = False
rollout_parameters["action_key"] = "expert_action"
for global_step in range(n_iters):
    logging_data.update({"mode": "training"})
    iter_start_time = time.time()
    ess_weight = entropy_coefficient * entropy_coefficient_decay**global_step
    # collect data
    env_rollout_start_time = time.time()
    with torch.no_grad():
        rollouts = parallel_rollout(
            envs=envs,
            num_parallel_envs=n_parallel_envs,
            rollout_params=rollout_parameters,
        )
    # rollouts = rollouts # [DEBUG]
    env_rollout_time = time.time() - env_rollout_start_time

    itr_training_start_time = time.time()
    train(rollouts)
    training_time_per_itr = time.time() - itr_training_start_time
    logging_data.update(
        {
            "itr_training_time": training_time_per_itr,
            "total_training_time": time.time() - start_time,
            "itr_time": time.time() - iter_start_time,
            "n_iter": global_step,
            "stage": "ppo" if bc_finished_flag else "bootstrap",
        }
    )

    if global_step % 2 == 0 and global_step > 0:
        torch.save(
            wrapped_actor_critic.state_dict(),
            model_dir / f"ppo_parallel_{global_step}.pt",
        )
        # evaluate
        bc_finished_flag = evaluate(n_eval_envs=n_eval_envs)
        if bc_finished_flag:
            rollout_parameters["action_key"] = "action"
        else:
            rollout_parameters["action_key"] = "expert_action"
        print_data(logging_data)


torch.save(wrapped_actor_critic.state_dict(), model_dir / f"actor_critic_{global_step}.pt")
