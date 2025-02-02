from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    def __init__(
        self,
        *,
        policy_network: torch.nn.Module,
        forward_preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        super().__init__()
        self._policy_network = policy_network
        self._forward_preprocess = forward_preprocess

    def _pre_forward(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        if self._forward_preprocess is not None:
            frame_data = self._forward_preprocess(frame_data)
        return frame_data

    @property
    def policy_network(self):
        return self._policy_network

    def forward(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        frame_data = self._pre_forward(frame_data)
        pred = self._policy_network(
            frame_data["observation"]["graph"].x,
            frame_data["observation"]["graph"].edge_index,
            frame_data["observation"]["graph"].batch,
        )
        probabilities = F.softmax(pred, dim=0).reshape(-1)
        dist = torch.distributions.Categorical(probabilities)
        action_index = dist.sample()
        log_prob = torch.log(probabilities.gather(0, action_index))
        frame_data["action"] = action_index
        frame_data["log_prob"] = log_prob
        return frame_data

    def forward_parallel(self, graph, batch, *, expert_actions=None):
        pred = self._policy_network(graph.x, graph.edge_index, graph.batch, masks=batch)
        batches = int(batch.max().item()) + 1
        actions = []
        log_probs = []
        entropies = []
        for index in range(batches):
            mask = batch == index
            probabilities = F.softmax(pred[mask], dim=0).reshape(-1)
            dist = torch.distributions.Categorical(probabilities)
            action_index = dist.sample() if expert_actions is None else expert_actions[index]
            log_prob = torch.log(probabilities.gather(0, action_index))
            entropy = dist.entropy()
            actions.append(action_index)
            log_probs.append(log_prob)
            entropies.append(entropy)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        return actions, log_probs, entropies

    __call__ = forward


class Critic(nn.Module):
    def __init__(
        self,
        *,
        value_network: torch.nn.Module,
        forward_preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        super().__init__()
        self._value_network = value_network
        self._forward_preprocess = forward_preprocess

    def _pre_forward(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        if self._forward_preprocess is not None:
            frame_data = self._forward_preprocess(frame_data)
        return frame_data

    @property
    def value_network(self):
        return self._value_network

    def forward(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        frame_data = self._pre_forward(frame_data)
        value = self._value_network(
            frame_data["observation"]["graph"].x,
            frame_data["observation"]["graph"].edge_index,
            frame_data["observation"]["graph"].batch,
        )
        frame_data.update({"value": value})
        return frame_data

    def forward_parallel(self, graph, batch):
        return self._value_network(
            graph.x,
            graph.edge_index,
            graph.batch,
            masks=batch,
        )

    __call__ = forward


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        *,
        forward_preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        forward_actor: bool = True,
        forward_critic: bool = True,
    ):
        super().__init__()
        self._actor = actor
        self._critic = critic
        self._forward_actor = forward_actor
        self._forward_critic = forward_critic
        self._forward_preprocess = forward_preprocess

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic

    def forward_actor(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        return self._actor(frame_data)

    def forward_critic(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        return self._critic(frame_data)

    def forward(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        frame_data = self._pre_forward(frame_data)
        if self._forward_actor:
            frame_data = self._actor(frame_data)
        if self._forward_critic:
            frame_data = self._critic(frame_data)
        return frame_data

    def _pre_forward(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        if self._forward_preprocess is not None:
            frame_data = self._forward_preprocess(frame_data)
        return frame_data

    def forward_parallel(self, graph, batch, *, expert_actions=None):
        action_index, log_prob, entropy = self.actor.forward_parallel(graph, batch, expert_actions=expert_actions)
        value = self.critic.forward_parallel(graph, batch)
        return action_index, log_prob, value, entropy

    __call__ = forward


class GAE:
    def __init__(self, *, gamma: float, lmbda: float):
        super().__init__()
        self._gamma = gamma
        self._lambda = lmbda

    def forward(self, rollout: list[dict[str, Any]]) -> list[dict[str, Any]]:
        advantage = 0
        for i in reversed(range(len(rollout))):
            reward = rollout[i]["next"]["reward"]["total_reward"]
            value = rollout[i]["value"]
            next_value = rollout[i]["next"]["value"]
            next_done = rollout[i]["next"]["done"]
            delta = reward + self._gamma * next_value * (1.0 - next_done) - value
            advantage = delta + self._gamma * self._lambda * (1.0 - next_done) * advantage
            rollout[i]["advantage"] = advantage
            rollout[i]["return"] = advantage + value
        return rollout

    __call__ = forward
