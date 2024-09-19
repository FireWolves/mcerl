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
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from rich.console import Console
from rich.table import Table
from rl.actor_critic import GAE, Actor, ActorCritic, Critic
from rl.network import GINPolicyNetwork, GINValueNetwork
from rl.utils import Sampler, to_graph
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from mcerl.env import Env
from mcerl.utils import (
    delta_time_reward_standardize,
    exploration_reward_rescale,
    reward_sum,
)

if __name__ == "__main__":
    # %%
    # load map
    map_path = "map/0.png"
    img = Image.open(map_path)
    img = ImageOps.grayscale(img)
    img = img.resize((300, 200))
    grid_map = np.array(img)
    grid_map[grid_map < 100] = 0
    grid_map[grid_map >= 100] = 255
    plt.imshow(grid_map, cmap="gray", vmin=0, vmax=255)

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
    cpp_env_log_level = log_level
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
    num_rays = 32

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

    # 地图的高度和宽度
    map_height, map_width = grid_map.shape

    # 一个frontier最多可以获得多少信息增益
    max_exploration_gain = ray_range**2 * pi / 2.0

    #########################
    # PPO参数
    #########################

    # gae权重
    lmbda = 0.95

    # discount factor
    gamma = 0.99

    # reward scale
    # tanh(x/a)
    a = 500

    # 1/(1+((x-x_star)/sigma)^(2*b))
    b = 1
    x_star = 0
    sigma = 100

    # clip范围
    clip_coefficient = 0.2

    # 最大gradient范数
    max_grad_norm = 0.5

    # ESS
    entropy_weight = 0.001

    # policy loss的权重
    policy_loss_weight = 1.0

    # value loss的权重
    value_loss_weight = 0.5

    #########################
    # 训练参数
    #########################

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 总共训练的次数
    n_iters = 1000

    # 每次训练的frame数(大约)
    n_frames_per_iter = 10000

    # 每次训练的并行环境数(num worker)
    n_parallel_envs = 32

    # 每次训练的epoch数, 即数据要被训练多少次
    n_epochs_per_iter = 5

    # 每次epoch的mini_batch大小
    n_frames_per_mini_batch = 500

    # 每个agent的最大步数
    max_steps_per_agent = 40

    # 每次训练所用到的总的环境数量
    n_envs = round(n_frames_per_iter / num_agents / max_steps_per_agent)

    # 每个epoch的frame数
    n_frames_per_epoch = n_frames_per_iter

    # 每个epoch的mini_batch数
    n_minibatches_per_epoch = n_frames_per_epoch // n_frames_per_mini_batch

    parameters = {k: v for k, v in locals().items() if k not in [*exclude_parameters, "exclude_parameters"]}
    console = Console()
    console.print(parameters)

    # %%
    # transform function
    # env_transform
    def env_transform(frame_data: dict[str, Any]) -> dict[str, Any]:
        """
        normalize position, exploration gain,etc.
        Note that we need to check if env is done
        """

        width = float(map_width)
        height = float(map_height)

        # normalize frontier position to [0,1] and exploration gain to [0,1]
        frame_data["observation"]["frontier_points"] = [
            (
                float(x) / width,
                float(y) / height,
                float(gain) / max_exploration_gain,
                np.tanh(float(distance) / float(ray_range)),
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
        device = torch.device("cuda") if device is None else device
        if "graph" in frame_data["observation"]:
            frame_data["observation"]["graph"] = frame_data["observation"]["graph"].to(device)

        return frame_data

    # %%
    # define policy

    policy_network = GINPolicyNetwork(dim_feature=6, dim_h=32)
    value_network = GINValueNetwork(dim_feature=6, dim_h=32)
    actor = Actor(policy_network=policy_network)
    critic = Critic(value_network=value_network)
    wrapped_actor_critic = ActorCritic(actor=actor, critic=critic, forward_preprocess=device_cast)
    wrapped_actor_critic = wrapped_actor_critic.to(device)
    value_estimator = GAE(gamma=gamma, lmbda=lmbda)
    optimizer = Adam(wrapped_actor_critic.parameters())

    # %%
    # load state dict
    # wrapped_actor_critic.load_state_dict(torch.load("ppo_parallel_100.pt"))

    # %%
    # parameters for env rollout

    rollout_parameters = {
        "grid_map": grid_map,
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
        "return_maps": True,
        "requires_grad": False,
    }
    console.print(rollout_parameters)

    # %%
    # single rollout
    # env = Env(
    #     log_level,
    #     cpp_env_log_path.as_posix(),
    # )
    # with torch.no_grad():
    #     single_rollouts = env.rollout(**rollout_parameters)
    # rollouts = single_rollouts

    # %%
    # parallel rollout

    def rollout_env(env, params):
        """在给定的环境中执行rollout。"""
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
                    console.print(f"环境 {futures[future]} 的rollout执行时发生错误: {e}")

        return results

    envs = [Env(cpp_env_log_level, cpp_env_log_path=cpp_env_log_path.as_posix()) for i in range(n_envs)]
    console.print(len(envs))

    # %%
    # with torch.no_grad():
    #     rollouts = parallel_rollout(
    #                 envs=envs,
    #                 num_parallel_envs=n_parallel_envs,
    #                 rollout_params=rollout_parameters,
    #             )

    # %%
    # for logging
    writer = SummaryWriter(tensorboard_dir / "ppo")
    writer.add_text("parameters", str(parameters))

    # %%
    # train loop

    console_logging_data = {}
    start_time = time.time()
    for _n_iter in range(n_iters):
        # collect data
        iter_start_time = time.time()
        data_collection_start_time = time.time()
        with torch.no_grad():
            env_rollout_start_time = time.time()
            rollouts = parallel_rollout(
                envs=envs,
                num_parallel_envs=n_parallel_envs,
                rollout_params=rollout_parameters,
            )
            env_rollout_time = time.time() - env_rollout_start_time

            data_post_process_start_time = time.time()
            # normalize rewards
            rollouts = [exploration_reward_rescale(rollout, a=a) for rollout in rollouts]
            rollouts = [delta_time_reward_standardize(rollout, b=b, sigma=sigma, x_star=x_star) for rollout in rollouts]
            rollouts = [reward_sum(rollout, gamma=gamma) for rollout in rollouts]
            data_post_process_time = time.time() - data_post_process_start_time

            gae_start_time = time.time()
            # compute GAE
            rollouts = [value_estimator(rollout) for rollout in rollouts]
            flattened_rollouts = [frame_data for rollout in rollouts for frame_data in rollout]
            actual_frames_per_iter = len(flattened_rollouts)
            gae_time = time.time() - gae_start_time
            # add to sampler

            graphs = []
            rewards = []
            values = []
            advantages = []
            log_probs = []
            returns = []
            total_times = []

            for frame_data in flattened_rollouts:
                graphs.append(frame_data["observation"]["graph"])
                rewards.append(frame_data["next"]["reward"]["total_reward"])
                values.append(frame_data["value"])
                advantages.append(frame_data["advantage"])
                log_probs.append(frame_data["log_prob"])
                returns.append(frame_data["return"])
                total_times.append(frame_data["info"]["total_time_step"])
            rewards = torch.tensor(rewards).to(device)
            values = torch.tensor(values).to(device)
            advantages = torch.tensor(advantages).to(device)
            log_probs = torch.tensor(log_probs).to(device)
            returns = torch.tensor(returns).to(device)
            total_times = torch.tensor(total_times).to(device)

            sampler = Sampler(
                batch_size=n_frames_per_mini_batch,
                length=len(flattened_rollouts),
                graphs=graphs,
                rewards=rewards,
                values=values,
                advantages=advantages,
                log_probs=log_probs,
                returns=returns,
                total_times=total_times,
            )
        data_collection_time = time.time() - data_collection_start_time

        training_start_time = time.time()
        clip_fraction = []
        for _n_epoch in range(n_epochs_per_iter):
            epoch_start_time = time.time()
            for _n_mini_batch in range(n_minibatches_per_epoch):
                mini_batch_start_time = time.time()

                sample_start_time = time.time()
                # sample data
                mini_batch_data = sampler.random_sample()
                graphs = mini_batch_data["graphs"].to(device)
                rewards = mini_batch_data["rewards"]
                values = mini_batch_data["values"].to(device).flatten()
                advantages = mini_batch_data["advantages"].to(device).flatten()
                prev_log_prob = mini_batch_data["log_probs"].to(device).flatten()
                returns = mini_batch_data["returns"].to(device).flatten()
                total_times = mini_batch_data["total_times"]
                frame_indices = mini_batch_data["frame_indices"].to(device)
                sample_time = time.time() - sample_start_time

                training_data_prepare_start_time = time.time()
                # get minibatch data and transform to tensor for training

                forward_start_time = time.time()
                new_action, new_log_probs, new_values, entropy = wrapped_actor_critic.forward_parallel(
                    graphs, frame_indices
                )
                new_log_probs = new_log_probs.to(device).flatten()
                new_values = new_values.to(device).flatten()
                forward_time = time.time() - forward_start_time

                data_prepare_time = time.time() - training_data_prepare_start_time

                loss_compute_start_time = time.time()
                # compute loss
                log_ratio = new_log_probs - prev_log_prob
                ratio = log_ratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction += [((ratio - 1.0).abs() > clip_coefficient).float().mean().item()]

                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coefficient, 1 + clip_coefficient)

                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()
                ess_loss = entropy.mean()
                loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss - entropy_weight * ess_loss
                loss_compute_time = time.time() - loss_compute_start_time

                optimizer_start_time = time.time()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapped_actor_critic.parameters(), max_grad_norm)
                optimizer.step()
                optimizer_time = time.time() - optimizer_start_time
                mini_batch_time = time.time() - mini_batch_start_time

            epoch_end_time = time.time() - epoch_start_time
            console_logging_data.update(
                {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "ESS": ess_loss.item(),
                    "old_approx_kl": old_approx_kl.item(),
                    "approx_kl": approx_kl.item(),
                    "clip_fraction": np.mean(clip_fraction),
                    "loss": loss.item(),
                    "loss_compute_time": loss_compute_time,
                    "optimizer_time": optimizer_time,
                    "sample_time": sample_time,
                    "forward_time": forward_time,
                    "data_prepare_time": data_prepare_time,
                    "mini_batch_time": mini_batch_time,
                    "data_collection_time": data_collection_time,
                    "env_rollout_time": env_rollout_time,
                    "data_post_process_time": data_post_process_time,
                    "gae_time": gae_time,
                    "mini_batch_size": f"{n_minibatches_per_epoch}/{n_frames_per_mini_batch}",
                    "n_iter": f"{_n_iter+1}/{n_iters}",
                    "n_epoch": f"{_n_epoch+1}/{n_epochs_per_iter}",
                    "frames_per_iter": f"{actual_frames_per_iter}/{n_frames_per_iter}",
                }
            )
            table = Table(title=f"PPO Training: {n_envs} envs in {n_parallel_envs} threads")
            table.add_column("Key", justify="left")
            table.add_column("Value", justify="left")
            for k, v in console_logging_data.items():
                table.add_row(k, str(v)[:16])
            console.print(table)

        training_time_per_itr = time.time() - training_start_time

        average_exploration_time = torch.mean(total_times.to(torch.float)).item()
        episode_reward_mean = torch.mean(returns).item()

        console_logging_data.update(
            {
                "training_time_per_itr": training_time_per_itr,
                "average_exploration_time": average_exploration_time,
                "episode_reward_mean": episode_reward_mean,
                "total_training_time": time.time() - start_time,
            }
        )
        global_step = _n_iter
        if global_step % 10 == 0 and global_step > 0:
            torch.save(
                wrapped_actor_critic.state_dict(),
                model_dir / f"ppo_parallel_{global_step}.pt",
            )
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/ESS", ess_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clip_fraction", np.mean(clip_fraction), global_step)
        writer.add_scalar("info/episode_reward_mean", episode_reward_mean, global_step)
        writer.add_scalar("info/average_episode_time", average_exploration_time, global_step)

        iter_time = time.time() - iter_start_time
        console_logging_data.update(
            {
                "iter_time": iter_time,
            }
        )

    torch.save(wrapped_actor_critic.state_dict(), model_dir / f"actor_critic_{global_step}.pt")
