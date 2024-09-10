from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from sympy import sequence
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
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool

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


# class DoubleGraphActorCritic(ActorCriticBase):
#     def __init__(
#         self,
#         *,
#         common_feature_dim_agent: int,
#         common_hidden_dim_agent: int,
#         common_activation_agent: str,
#         common_feature_dim_target: int,
#         common_hidden_dim_target: int,
#         common_activation_target: str,
#         common_num_attention_heads: int,
#         common_num_transformer_layers: int,
#         action_head_hidden_dim: int,
#         action_head_output_dim: int,
#         action_head_activation: str,
#         value_head_hidden_dim: int,
#         value_head_activation: str,
#     ):
#         super().__init__()
#         self.agent_encoder = GINNet(
#             input_dim=common_feature_dim_agent,
#             hidden_dim=common_hidden_dim_agent,
#             activation=common_activation_agent,
#         )
#         self.target_encoder = GINNet(
#             input_dim=common_feature_dim_target,
#             hidden_dim=common_hidden_dim_target,
#             activation=common_activation_target,
#         )

#         transformer_layer = TransformerEncoderLayer(
#             d_model=common_hidden_dim_target * 3, nhead=common_num_attention_heads
#         )
#         self.transformer_encoder = TransformerEncoder(
#             transformer_layer, num_layers=common_num_transformer_layers
#         )

#         self.action_head = nn.Sequential(
#             nn.Linear(common_hidden_dim_agent * 3, action_head_hidden_dim),
#             get_activation_fn(action_head_activation)(),
#             nn.Linear(action_head_hidden_dim, action_head_output_dim),
#         )
#         self.value_head = nn.Sequential(
#             nn.Linear(common_hidden_dim_target * 3, value_head_hidden_dim),
#             get_activation_fn(value_head_activation)(),
#             nn.Linear(value_head_hidden_dim, 1),
#         )

#     def actor(self, agent_graph, target_graph, return_logits=True):
#         # 编码 agent graph
#         agent_features = self.agent_encoder(agent_graph)

#         # 编码 target graph
#         target_features = self.target_encoder(target_graph)

#         # 将 agent 和 target 特征拼接
#         combined_features = torch.cat([agent_features, target_features], dim=-1)

#         # 通过 transformer 层
#         transformed_features = self.transformer_encoder(combined_features)

#         # 提取 agent 相关的特征
#         agent_transformed = transformed_features[:, : agent_features.size(1), :]

#         # 平均池化得到全局特征
#         global_agent_features = torch.mean(agent_transformed, dim=1)

#         # 通过 action head 得到动作概率分布
#         logits = self.action_head(global_agent_features)
#         probs = F.softmax(logits, dim=-1)
#         action = torch.multinomial(probs, num_samples=1)
#         log_prob = F.log_softmax(logits, dim=-1).gather(1, action)
#         if return_logits:
#             return action.squeeze(-1), log_prob.squeeze(-1), logits
#         return action.squeeze(-1), log_prob.squeeze(-1)

#     def critic_operator(self, agent_graph, target_graph):
#         # 编码 agent graph
#         agent_features = self.agent_encoder(agent_graph)

#         # 编码 target graph
#         target_features = self.target_encoder(target_graph)

#         # 将 agent 和 target 特征拼接
#         combined_features = torch.cat([agent_features, target_features], dim=-1)

#         # 通过 transformer 层
#         transformed_features = self.transformer_encoder(combined_features)

#         # 提取 target 相关的特征
#         target_transformed = transformed_features[:, agent_features.size(1) :, :]

#         # 平均池化得到全局特征
#         global_target_features = torch.mean(target_transformed, dim=1)

#         # 通过 value head 得到价值估计
#         value = self.value_head(global_target_features)

#         return value.squeeze(-1)


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
    ):
        super().__init__()
        self._policy_network = policy_network

    def _pre_forward(self, frame_data):
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
            action_index = torch.multinomial(probabilities.squeeze(), 1)
            log_prob = torch.log(probabilities[action_index])
            frame_data["action"] = action_index
            frame_data["log_prob"] = log_prob
        return frame_data
