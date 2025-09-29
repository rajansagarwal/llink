from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, dropout_p: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(3072, 2048),
        )
        self.ln = nn.LayerNorm(2048)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(z))


def load_projector_from_state(state_dict: Dict) -> nn.Module:
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    projector = Projector()
    projector.load_state_dict(state_dict)
    return projector


class TokenExpander(nn.Module):
    def __init__(self, d: int = 2048, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d * k),
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        expanded = self.net(v).view(v.size(0), self.k, -1)
        return self.ln(expanded)


class ForeignScale(nn.Module):
    def __init__(self, hidden_size: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(vector, p=2, dim=-1)
        normalized = self.ln(normalized)
        return normalized * self.scale


class IdentityGatedAdapter(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int = 2048, g0: float = 0.3) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.g = nn.Parameter(torch.tensor(g0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.w2(F.gelu(self.w1(self.ln(inputs))))
        gate = torch.tanh(self.g).clamp(0, 0.4)
        return inputs + gate * residual
