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
class NcpConfigMultipleStep:
    ncp_units: int
    embed_dim: int
    num_layers: int
    embed_pdrop: float
    blocks_pdrop: float
    ncp_layer_norm: bool  # Apply layer normalization after each NCP block
    pos_emb: bool  # Apply positional embedding to the input data using step index
    max_blocks: int
    tokens_per_block: int

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class WorldModelNcpMultipleStep(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: NcpConfigMultipleStep) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)

        # Embedding
        # TODO should I use the all_but_last_obs_tokens_pattern in head_observations as in Transformer?
        # all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        # all_but_last_obs_tokens_pattern[-2] = 0
        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)  # eg. [1, 1, 1, 1, 1, 1, 0, 1]
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)         # eg. [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern                            # eg. [1, 1, 1, 1, 1, 1, 1, 0]

        # TODO WorldModelNcpMultipleStep - positional embedding yes or no?
        if self.config.pos_emb:
            self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        # This embeds actions and observations with different embedder
        self.embedder = Embedder(
            max_blocks=config.max_blocks,  # TODO What does max_blocks do in the Embedder (extends Slicer)?
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([
                nn.Embedding(act_vocab_size, config.embed_dim),
                nn.Embedding(obs_vocab_size, config.embed_dim)
            ])
        )

        # NCP layers
        self.ncp_layers = nn.ModuleList([
            NcpBlock(config.embed_dim, config.ncp_units, config.blocks_pdrop, config.ncp_layer_norm,
                     return_sequences=True)  # (i < config.num_layers - 1))
             for _ in range(config.num_layers)
        ])

        # self.ncp_layer_wiring = AutoNCP(config.ncp_units, config.embed_dim)
        # self.ncp_layer_1 = CfC(config.embed_dim, self.ncp_layer_wiring)  # 16 for observation + 1 for action

        # Heads
        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,  # eg. [1, 1, 1, 1, 1, 1, 0, 1]
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,  # eg. [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,  # eg. [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        self.apply(init_weights_wm_ncp)

    def __repr__(self) -> str:
        return "world_model_ncp_multiple_step"

    def forward(self, tokens: torch.IntTensor, hidden_state: Optional[Tuple] = None, step_idx: int = 0) -> WorldModelOutput:
        # TODO is step_idx used the right way. No it embeds it's position in the frames sequence * 16.
        #  I think it should be the position in the sequence of steps of single frame, not all frames.
        assert len(tokens.size()) == 2
        tokens = tokens.to(torch.int64)
        num_steps = tokens.size(1)  # (B, T)
        prev_steps = step_idx
        assert num_steps == 1  # For now, we can do only one step at a time as it is a recurrent model

        if self.config.pos_emb:
            pos_emb = self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))
        else:
            pos_emb = torch.zeros((num_steps, self.config.embed_dim), device=tokens.device)

        x = self.embedder(tokens, num_steps, prev_steps) + pos_emb
        # x = rearrange(x, 'b t s e -> b t (s e)')
        x = self.drop(x)
        new_hidden_states = []
        for i, layer in enumerate(self.ncp_layers):
            x, h_state = layer(x, hidden_state[i] if hidden_state else None)
            new_hidden_states.append(h_state)

        # logits_observations = self.head_observations(x).reshape(tokens.size(0), tokens.size(1), -1)
        # logits_rewards = self.head_rewards(x)
        # logits_ends = self.head_ends(x)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, new_hidden_states)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        with torch.no_grad():
            tokenizer_out = tokenizer.encode(batch['observations'], should_preprocess=True)  # (B, L, K)

        obs_tokens = tokenizer_out.tokens
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        # tokens = torch.cat((obs_tokens, act_tokens), dim=2)  # (B, L, K)

        outputs = []
        hidden_state = None
        for i in range(tokens.size(1)):
            # TODO Should I really use step_idx=i here? Step_idx will be up to 340, but in real world model it will be only up to 16.
            #  like: (i % 16) + 1
            output = self(tokens[:, i:i+1], hidden_state, i)  # forward() is called here
            outputs.append(output)
            hidden_state = output.hidden_state

        outputs = WorldModelOutput(
            output_sequence=torch.cat([output.output_sequence for output in outputs], dim=1),
            logits_observations=torch.cat([output.logits_observations for output in outputs], dim=1),
            logits_rewards=torch.cat([output.logits_rewards for output in outputs], dim=1),
            logits_ends=torch.cat([output.logits_ends for output in outputs], dim=1),
            hidden_state=outputs[-1].hidden_state
        )

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(tokenizer_out.tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        # logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t e o -> (b t e) o')
        # loss_obs = F.cross_entropy(logits_observations, labels_observations)
        # loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        # loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)


    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)

    # def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
    #     mask_fill = torch.logical_not(mask_padding)
    #     labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100)[:, 1:], 'b t k -> b (t k)')
    #     labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
    #     labels_ends = ends.masked_fill(mask_fill, -100)
    #     # return labels_observations, labels_rewards, labels_ends
    #     return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)


class NcpBlock(nn.Module):
    def __init__(self, embed_dim: int, ncp_units: int, pdrop: float, ncp_layer_norm: Optional[bool] = True, return_sequences=False) -> None:
        super().__init__()
        self.wiring = AutoNCP(ncp_units, embed_dim)
        self.ncp = CfC(embed_dim, self.wiring, return_sequences=return_sequences)
        self.drop = nn.Dropout(pdrop)
        # TODO layer norm ?
        self.norm = nn.LayerNorm(embed_dim) if ncp_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor, hidden_state=None) -> [torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        x, h = self.ncp(x, hidden_state)
        x = self.drop(x)
        return x, h
