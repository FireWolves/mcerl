from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    BatchNorm1d,
    Linear,
    Module,
    ReLU,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.modules import activation
from torch_geometric.nn import GINConv, global_add_pool

Activation = Callable[..., Module]


def get_activation_fn(act: str) -> Activation:
    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_lc:
        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        return getattr(activation, act_name)
    msg = f"Cannot find activation function for string <{act}>"
    raise ValueError(msg)


class ActorCriticBase(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @property
    def actor(self):
        raise NotImplementedError

    @property
    def critic(self):
        raise NotImplementedError


class ActorCritic(ActorCriticBase):
    def __init__(
        self,
    ):
        super().__init__()

    def actor(self, **kwargs):
        raise NotImplementedError

    def critic(self, **kwargs):
        raise NotImplementedError


class GINNet(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        activation: str,
    ):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation_fn(activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation_fn(activation)(),
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation_fn(activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation_fn(activation)(),
        )
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn2)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        return torch.cat([global_add_pool(h, batch=batch) for h in [h1, h2, h3]], dim=1)


class GINPolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_h,
        num_transformer_layers=1,
        transformer_heads=8,
    ):
        super().__init__()
        # Existing GIN layers
        nn1 = Sequential(
            Linear(5, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
        )
        self.conv1 = GINConv(nn1)
        nn2 = Sequential(
            Linear(dim_h, dim_h),
            BatchNorm1d(dim_h),
            ReLU(),
            Linear(dim_h, dim_h),
            ReLU(),
        )
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn2)

        # Transformer layer
        transformer_layer = TransformerEncoderLayer(
            d_model=dim_h * 3, nhead=transformer_heads
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )

        # Linear layers
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, 1)  # Output 1 value for regression

    def forward(self, x, edge_index, batch):
        # GIN layers
        h1 = self.conv1(x, edge_index).relu()

        h2 = self.conv2(h1, edge_index).relu()

        h3 = self.conv3(h2, edge_index).relu()

        h = torch.cat([global_add_pool(h, batch) for h in [h1, h2, h3]], dim=1)
        # Transformer layer
        h_transformed = self.transformer_encoder(h.unsqueeze(1)).squeeze(
            1
        )  # Assuming batch-first

        # Linear layers
        h = self.lin1(h_transformed).relu()
        h = F.dropout(h, p=0.5, training=self.training)

        return self.lin2(h)


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
            if(probabilities.numel() == 1):
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
