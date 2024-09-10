from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


def build_graphs(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Builds a graph representation for each frame in the trajectory.
    Args:
        trajectory (list[dict[str, Any]]): A list of dictionaries containing frame data.
    Returns:
        list[dict[str, Any]]: A list of dictionaries containing frame data with the graph representation added.
    """
    for i in range(len(trajectory)):
        trajectory[i] = to_graph(trajectory[i])
        trajectory[i]["next"] = to_graph(trajectory[i]["next"])
    return trajectory


def to_graph(frame_data):
    """
    Converts frame data into a graph representation.
    Args:
        frame_data (dict): A dictionary containing frame data.
    Returns:
        dict: The updated frame data dictionary with the graph representation added.
    """

    frame_frontier_points = torch.tensor(frame_data["observation"]["frontier_points"])
    frame_agent_poses = torch.tensor(frame_data["observation"]["pos"])
    frame_target_poses = torch.tensor(frame_data["observation"]["target_pos"])
    xs = []
    for frontier_point in frame_frontier_points:
        x = torch.cat(
            [
                torch.ones(frame_agent_poses.shape[0], 1) * frontier_point[-1],
                frame_agent_poses,
                frame_target_poses,
            ],
            dim=1,
        )
        x[0, -2:] = frontier_point[:2]
        xs.append(x)
    edge_index = torch.arange(frame_agent_poses.shape[0]).repeat(2, 1)
    edge_index[0] = 0
    edge_index = edge_index[..., 1:]
    edge_index = to_undirected(edge_index)
    data = DataLoader(
        [Data(x=x, edge_index=edge_index) for x in xs],
        batch_size=frame_frontier_points.shape[0],
    )
    # torch_geometric only supports creating batched data from a DataLoader, maybe there is a better way to do this but I couldn't find it
    batch_data = next(iter(data))
    frame_data["observation"]["graph"] = batch_data
    return frame_data
