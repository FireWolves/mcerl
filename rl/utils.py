from __future__ import annotations

import copy
import logging
import random
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


def to_graph(frame_data, *, device: torch.device | None = None) -> dict[str, Any]:
    """
    Converts frame data into a graph representation.
    Args:
        frame_data (dict): A dictionary containing frame data.
    Returns:
        dict: The updated frame data dictionary with the graph representation added.
    """
    if (
        frame_data["observation"]["frontier_points"] is None
        or len(frame_data["observation"]["frontier_points"]) == 0
    ):
        logging.warning("No frontier points found in frame data.")
        return frame_data

    frame_frontier_points = torch.tensor(frame_data["observation"]["frontier_points"])
    frame_agent_poses = torch.tensor(frame_data["observation"]["pos"])
    frame_target_poses = torch.tensor(frame_data["observation"]["target_pos"])
    xs = []
    for frontier_point in frame_frontier_points:
        x = torch.cat(
            [
                torch.ones(frame_agent_poses.shape[0], 1) * frontier_point[-2],
                torch.ones(frame_agent_poses.shape[0], 1) * frontier_point[-1],
                frame_agent_poses,
                frame_target_poses,
            ],
            dim=1,
        )
        x[0, -2:] = frontier_point[:2]
        xs.append(x)
    edge_index = torch.arange(frame_agent_poses.shape[0]).repeat(2, 1)
    edge_index_before = edge_index.clone()
    edge_index[0] = 0
    edge_index = edge_index[..., 1:]
    edge_index = to_undirected(edge_index)
    logging.debug(
        "edge_index before: %s, edge_index after: %s, xs: %s, batch_size: %s",
        edge_index_before,
        edge_index,
        xs,
        frame_frontier_points.shape[0],
    )
    data = DataLoader(
        [Data(x=x, edge_index=edge_index) for x in xs],
        batch_size=frame_frontier_points.shape[0],
    )
    # torch_geometric only supports creating batched data from a DataLoader, maybe there is a better way to do this but I couldn't find it
    batch_data = next(iter(data))
    batch_data.x = batch_data.x.to(device)
    batch_data.edge_index = batch_data.edge_index.to(device)
    batch_data.batch = batch_data.batch.to(device)
    batch_data = batch_data.to(device)
    frame_data["observation"]["graph"] = batch_data
    return frame_data


class Sampler:
    def __init__(
        self,
        batch_size: int,
        length: int,
        **kwargs,
    ) -> None:
        self.__dict__.update(kwargs)
        self._keys = list(kwargs.keys())
        self._keys.remove("graphs")
        self._length = length
        self._batch_size = batch_size

    def random_sample(self) -> dict[str, Any]:
        indices = random.sample(range(self._length), self._batch_size)
        graphs = [self.graphs[i] for i in indices]  # type: ignore  # noqa: PGH003
        frame_indices = None
        sampled_graphs = []
        for graph in graphs:
            if frame_indices is None:
                frame_indices = [torch.zeros(graph.num_graphs)]
            else:
                frame_indices.append(
                    torch.ones(graph.num_graphs) * (frame_indices[-1].max() + 1)
                )
            for i in range(graph.num_graphs):
                sampled_graphs.append(graph[i])
        frame_indices = torch.cat(frame_indices)  # type: ignore  # noqa: PGH003
        sampled_graphs = DataLoader(
            sampled_graphs,
            batch_size=len(sampled_graphs),
        )
        sampled_graphs = next(iter(sampled_graphs))

        return {
            **{key: self.__dict__[key][indices] for key in self._keys},
            "graphs": sampled_graphs,
            "frame_indices": frame_indices,
        }


def check_dict_struct(d: dict[str, Any], prefix: str = "") -> list:
    flattened_keys = []
    for k, v in d.items():
        if isinstance(v, dict):
            flattened_keys.extend(check_dict_struct(v, prefix + k + "."))
        else:
            flattened_keys.append(prefix + k)
    return flattened_keys

