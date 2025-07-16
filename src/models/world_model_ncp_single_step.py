from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from utils import LossWithIntermediateLosses, init_weights_wm_ncp
from ncps.torch import CfC
from ncps.wirings import AutoNCP


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    hidden_state: List[torch.FloatTensor]


@dataclass
class NcpConfigSingleStep:
    ncp_units: int
    embed_dim: int
    num_layers: int
    embed_pdrop: float
    blocks_pdrop: float
    ncp_layer_norm: bool
    max_blocks: int
    tokens_per_block: int


class WorldModelNcpSingleStep(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: NcpConfigSingleStep) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)

        # Embedding
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        # This embeds actions and observations with different embedder
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([
                nn.Embedding(act_vocab_size, config.embed_dim),
                nn.Embedding(obs_vocab_size, config.embed_dim)
            ])
        )

        # NCP layers
        self.ncp_layers = nn.ModuleList([
            NcpBlock(config.embed_dim * 17, config.ncp_units, config.blocks_pdrop, config.ncp_layer_norm,
                     return_sequences=True)  # (i < config.num_layers - 1))
             for i in range(config.num_layers)
        ])

        # Heads
        self.head_observations = nn.Sequential(
            nn.Linear(config.embed_dim * 17, config.embed_dim * 17),
            nn.ReLU(),
            nn.Linear(config.embed_dim * 17, obs_vocab_size * 16)
        )

        self.head_rewards = nn.Sequential(
            nn.Linear(config.embed_dim * 17, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 3)
        )

        self.head_ends = nn.Sequential(
            nn.Linear(config.embed_dim * 17, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 2)
        )

        self.apply(init_weights_wm_ncp)

    def __repr__(self) -> str:
        return "world_model_ncp_single_step"

    def forward(self, tokens: torch.IntTensor, hidden_state: Optional[Tuple] = None) -> WorldModelOutput:
        tokens = tokens.to(torch.int64)
        num_steps = tokens.size(2)  # (B, T)
        prev_steps = 0

        assert len(tokens.size()) == 3
        assert tokens.size(2) == 17  # 16 for observation + 1 for action

        x = self.embedder(tokens, num_steps, prev_steps)
        x = rearrange(x, 'b t s e -> b t (s e)')
        x = self.drop(x)
        new_hidden_states = []
        for i, layer in enumerate(self.ncp_layers):
            x, h_state = layer(x, hidden_state[i] if hidden_state else None)
            new_hidden_states.append(h_state)

        logits_observations = self.head_observations(x).reshape(tokens.size(0), tokens.size(1), 16, -1)
        logits_rewards = self.head_rewards(x)
        logits_ends = self.head_ends(x)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, new_hidden_states)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        with torch.no_grad():
            tokenizer_out = tokenizer.encode(batch['observations'], should_preprocess=True)  # (B, L, K)

        obs_tokens = tokenizer_out.tokens
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = torch.cat((obs_tokens, act_tokens), dim=2)  # (B, L, K)

        outputs = self(tokens)  # forward() is called here

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(tokenizer_out.tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t e o -> (b t e) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100)[:, 1:], 'b t k -> b (t k)')
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)


class NcpBlock(nn.Module):
    def __init__(self, embed_dim: int, ncp_units: int, pdrop: float, ncp_layer_norm: Optional[bool] = True, return_sequences=False) -> None:
        super().__init__()
        self.wiring = AutoNCP(ncp_units, embed_dim)
        self.ncp = CfC(embed_dim, self.wiring, return_sequences=return_sequences)
        self.drop = nn.Dropout(pdrop)
        self.norm = nn.LayerNorm(embed_dim) if ncp_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor, hidden_state=None) -> [torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        x, h = self.ncp(x, hidden_state)
        x = self.drop(x)
        return x, h
