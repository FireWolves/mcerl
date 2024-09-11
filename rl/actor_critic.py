from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def actor(self, **kwargs):
        raise NotImplementedError

    def critic(self, **kwargs):
        raise NotImplementedError


class Actor:
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
        with torch.no_grad():
            frame_data = self._pre_forward(frame_data)
            pred = self._policy_network(
                frame_data["observation"]["graph"].x,
                frame_data["observation"]["graph"].edge_index,
                frame_data["observation"]["graph"].batch,
            )
            probabilities = F.softmax(pred, dim=0)
            if probabilities.numel() == 1:
                action_index = torch.tensor([0])
                log_prob = torch.tensor([0.0])
            else:
                action_index = torch.multinomial(probabilities.squeeze(), 1)
                log_prob = torch.log(probabilities[action_index])
            frame_data["action"] = action_index
            frame_data["log_prob"] = log_prob
        return frame_data

    __call__ = forward


class Critic:
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
        with torch.no_grad():
            frame_data = self._pre_forward(frame_data)
            value = self._value_network(
                frame_data["observation"]["graph"].x,
                frame_data["observation"]["graph"].edge_index,
                frame_data["observation"]["graph"].batch,
            )
            frame_data["value"] = value
        return frame_data

    __call__ = forward
