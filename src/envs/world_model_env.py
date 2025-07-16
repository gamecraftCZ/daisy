import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision

from models.world_model_dummy import WorldModelDummy
from models.world_model_ncp_multiple_step import WorldModelNcpMultipleStep
from models.world_model_ncp_single_step import WorldModelNcpSingleStep
from models.world_model_transformer import WorldModelTransformer


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm_hidden_state, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        # Initialize keys_values_wm_hidden_state with empty hidden state for the self.world_model

        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        if isinstance(self.world_model, WorldModelTransformer):
            self.keys_values_wm_hidden_state = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
            outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm_hidden_state)
            return outputs_wm.output_sequence  # (B, K, E)

        elif isinstance(self.world_model, WorldModelNcpSingleStep):
            tokens = torch.cat((obs_tokens.unsqueeze(1), torch.zeros(obs_tokens.size(0), 1, 1, device=self.device)), dim=2)  # (B, L, K)
            outputs_wm = self.world_model(tokens)
            self.keys_values_wm_hidden_state = outputs_wm.hidden_state  # default hidden state is None
            return outputs_wm.output_sequence  # (B, K, E)

        elif isinstance(self.world_model, WorldModelNcpMultipleStep):
            tokens = torch.zeros(n, 1, device=self.device)
            outputs_wm = self.world_model(tokens)
            self.keys_values_wm_hidden_state = outputs_wm.hidden_state  # default hidden state is None
            return outputs_wm.output_sequence  # (B, K, E)

        elif isinstance(self.world_model, WorldModelDummy):
            self.keys_values_wm_hidden_state = None
            return self.world_model.reset()

        else:
            raise NotImplementedError(f"World model of type {type(self.world_model)} is not supported by world_model_env.")

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True):
        if getattr(self.world_model, "transformer", None):
            assert self.keys_values_wm_hidden_state is not None
        assert self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if isinstance(self.world_model, WorldModelTransformer):
            if self.keys_values_wm_hidden_state.size + num_passes > self.world_model.config.max_tokens:
                _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        action_token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        action_token = action_token.reshape(-1, 1).to(self.device)  # (B, 1)

        if isinstance(self.world_model, WorldModelTransformer):
            for k in range(num_passes):  # assumption that there is only one action token.
                outputs_wm = self.world_model(action_token, past_keys_values=self.keys_values_wm_hidden_state)
                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

                if k < self.num_observations_tokens:
                    action_token = Categorical(logits=outputs_wm.logits_observations).sample()
                    obs_tokens.append(action_token)

        elif isinstance(self.world_model, WorldModelNcpSingleStep):
            tokens = torch.unsqueeze(torch.cat((self.obs_tokens, action_token), dim=1), dim=1)  # (B, L, K)
            outputs_wm = self.world_model(tokens, self.keys_values_wm_hidden_state)
            self.keys_values_wm_hidden_state = outputs_wm.hidden_state

            reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
            done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            obs_logits = outputs_wm.logits_observations.reshape(outputs_wm.logits_observations.size(0), -1, self.world_model.obs_vocab_size)
            obs_tokens = Categorical(logits=obs_logits).sample()

        elif isinstance(self.world_model, WorldModelNcpMultipleStep):
            for k in range(num_passes):  # assumption that there is only one action token.
                outputs_wm = self.world_model(action_token, hidden_state=self.keys_values_wm_hidden_state, step_idx=k+16)
                self.keys_values_wm_hidden_state = outputs_wm.hidden_state
                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

                if k < self.num_observations_tokens:
                    action_token = Categorical(logits=outputs_wm.logits_observations).sample()
                    obs_tokens.append(action_token)


        elif isinstance(self.world_model, WorldModelDummy):
            outputs_wm = self.world_model(action_token)
            reward = outputs_wm.logits_rewards.cpu().numpy()  # not logits even though it is called logits_rewards
            done = outputs_wm.logits_ends.cpu().numpy().astype(bool)  # not logits even though it is called logits_ends
            obs_tokens = outputs_wm.logits_observations  # not logits even though it is called logits_observations

        else:
            raise NotImplementedError(f"World model of type {type(self.world_model)} is not supported by world_model_env.")

        # output_sequence = torch.cat(output_sequence, dim=1)  # (B, 1 + K, E)
        if isinstance(self.world_model, WorldModelTransformer):
            self.obs_tokens = torch.cat(obs_tokens, dim=1)   # (B, K)

        if isinstance(self.world_model, WorldModelNcpSingleStep):
            self.obs_tokens = obs_tokens

        # DONE elif isinstance(self.world_model, WorldModelNcpMultipleStep):
        if isinstance(self.world_model, WorldModelNcpMultipleStep):
            # Concat multiple model steps into one frame observation (B, K)
            self.obs_tokens = torch.cat(obs_tokens, dim=1)

        if isinstance(self.world_model, WorldModelDummy):
            self.obs_tokens = obs_tokens

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
