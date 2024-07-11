from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class TokenizerCriticConfig:
    hidden_layer_size: int
    num_hidden_layers: int
    input_size: int

class TokenizerCritic(nn.Module):
    def __init__(self, config: TokenizerCriticConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        for i in range(config.num_hidden_layers):
            self.layers.append(nn.Linear(config.input_size if i == 0 else config.hidden_layer_size, config.hidden_layer_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(config.hidden_layer_size if config.num_hidden_layers > 0 else config.input_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
