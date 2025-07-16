from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gym.wrappers import TransformObservation
from hydra.utils import instantiate

from envs import SingleProcessEnv
from .kv_caching import KeysValues
from .tokenizer import Tokenizer


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class WorldModelDummy(nn.Module):
    def __init__(self, tokenizer: Tokenizer, obs_vocab_size: int, act_vocab_size: int, config, env_count, device) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        def create_env(cfg_env):
            def env_fn(*args, **kwargs):
                cfg = {**cfg_env}
                del cfg["noise_std"]
                env = partial(instantiate, config=cfg)(*args, **kwargs)
                env = TransformObservation(env, lambda obs: np.clip(
                    (obs + np.random.randn(*obs.shape) * 255. * cfg_env.noise_std)
                    , 0, 255.)
                                           ) if cfg_env.noise_std else env
                return env
            return SingleProcessEnv(env_fn)

        self.envs = [create_env(config.env) for env in range(env_count)]

    def __repr__(self) -> str:
        return "world_model_dummy"

    @torch.no_grad()
    def reset(self):
        obs = torch.stack(
            [torch.squeeze(torch.FloatTensor(env.reset()).div(255), dim=0) for i, env in enumerate(self.envs)]
        ).permute(0, 3, 1, 2).to(self.device).contiguous()
        encoded = self.tokenizer.encode(obs, should_preprocess=True)
        z_q = encoded.z_quantized
        return z_q

    @torch.no_grad()
    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:
        obs = []
        rewards = []
        ends = []
        for i in range(len(tokens)):
            env = self.envs[i]
            _obs, _reward, _done, _ = env.step(tokens[i])
            obs.append(torch.squeeze(torch.FloatTensor(_obs).div(255), dim=0).permute(2, 0, 1))
            rewards.append(torch.FloatTensor(_reward))
            ends.append(torch.FloatTensor(_done))

        obs = torch.stack(obs).to(self.device).contiguous()
        encoded = self.tokenizer.encode(obs, should_preprocess=True)
        tokens = encoded.tokens

        rewards = torch.stack(rewards).squeeze(dim=-1).sign().to(self.device)  # Rewards clipped to {-1, 0, 1}
        ends = torch.stack(ends).squeeze(dim=-1).to(self.device)

        return WorldModelOutput(torch.FloatTensor([]), tokens, rewards, ends)
