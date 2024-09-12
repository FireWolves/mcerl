import torch
from torch import nn
from utils import Sampler


# class PPO:
#     def __init__(
#         self, *, mini_batch: int, num_iters: int, num_epochs: int, **kwargs
#     ) -> None:
#         self._mini_batch = mini_batch
#         self._num_iters = num_iters
#         self._num_epochs = num_epochs
#         self.__dict__.update(kwargs)

#     def train(self, rollout: list[dict[str, Any]]) -> None:

#         for minibatch_rollout in sampler:

