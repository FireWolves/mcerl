from __future__ import annotations

from torch import nn


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


class GNNActorCritic(ActorCriticBase):
    def __init__(
        self,
    ):
        super().__init__()
        self._actor_network=

    @property
    def actor(self):
        raise NotImplementedError

    @property
    def critic(self):
        raise NotImplementedError
