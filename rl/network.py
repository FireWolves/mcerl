from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
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


class GINPolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_feature: int,
        dim_h,
        num_transformer_layers=1,
        transformer_heads=8,
    ):
        super().__init__()
        # Existing GIN layers
        nn1 = Sequential(
            Linear(dim_feature, dim_h),
            BatchNorm1d(dim_h),
            ReLU(),
            Linear(dim_h, dim_h),
            ReLU(),
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
            d_model=dim_h * 3, nhead=transformer_heads, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )

        # Linear layers
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, 1)  # Output 1 value for regression

    def forward(self, x, edge_index, batch, *, masks=None):
        # x:nodes of all targets:[n_agents*n_targets,n_node_features ]
        # edge_index: [2, 2 * n_targets(n_subgraphs)*n_agents(n_nodes_in_subgraph)]
        # batch: [n_subgraphs]
        # GIN layers

        h1 = self.conv1(x, edge_index).relu()

        h2 = self.conv2(h1, edge_index).relu()

        h3 = self.conv3(h2, edge_index).relu()
        h = torch.cat([global_add_pool(h, batch) for h in [h1, h2, h3]], dim=1)

        # Transformer layer
        h_transformed = self.transformer_encoder(h.unsqueeze(1))  # self-attention
        h_transformed = torch.mean(h_transformed, dim=1)
        # Linear layers
        h = self.lin1(h_transformed).relu()
        h = F.dropout(h, p=0.5, training=self.training)

        return self.lin2(h)


class GINValueNetwork(torch.nn.Module):
    def __init__(
        self, dim_feature, dim_h, num_transformer_layers=1, transformer_heads=8
    ):
        super().__init__()
        nn1 = Sequential(
            Linear(dim_feature, dim_h),
            BatchNorm1d(dim_h),
            ReLU(),
            Linear(dim_h, dim_h),
            ReLU(),
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
        transformer_layer = TransformerEncoderLayer(
            d_model=dim_h * 3, nhead=transformer_heads, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )
        self.lin1 = Linear(dim_h * 3, dim_h)
        self.lin2 = Linear(dim_h, 1)  # Output 1 value for regression

    def forward(self, x, edge_index, batch, *, masks=None):
        h1 = self.conv1(x, edge_index).relu()
        h2 = self.conv2(h1, edge_index).relu()
        h3 = self.conv3(h2, edge_index).relu()
        h = torch.cat([global_add_pool(h, batch) for h in [h1, h2, h3]], dim=1)

        if masks is not None:
            h = torch.nested.as_nested_tensor(
                [h[masks == i] for i in range(int(masks.max().item()) + 1)]
            )
            if self.training:
                h = torch.nested.to_padded_tensor(h, padding=0.0)

        if len(h.shape) < 3:
            h = h.unsqueeze(0)
        h_transformed = self.transformer_encoder(h)  # cross-attention
        h_transformed = torch.mean(h_transformed, dim=1)
        # Linear layers
        h = self.lin1(h_transformed).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)
