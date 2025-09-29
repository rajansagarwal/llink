from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, p: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(3072, 2048),
        )
        self.ln = nn.LayerNorm(2048)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(z))


def load_projector_from_state(sd: Dict) -> nn.Module:
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if {"net.0.weight", "net.3.weight"}.issubset(sd):
        w0, b0 = sd["net.0.weight"], sd["net.0.bias"]
        w1, b1 = sd["net.3.weight"], sd["net.3.bias"]
        lnw, lnb = sd["ln.weight"], sd["ln.bias"]

        class _P(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w0 = nn.Parameter(w0)
                self.b0 = nn.Parameter(b0)
                self.w1 = nn.Parameter(w1)
                self.b1 = nn.Parameter(b1)
                self.lnw = nn.Parameter(lnw)
                self.lnb = nn.Parameter(lnb)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                h = F.linear(z, self.w0, self.b0)
                h = F.gelu(h)
                h = F.linear(h, self.w1, self.b1)
                return F.layer_norm(h, self.lnw.shape, self.lnw, self.lnb)

        return _P()

    proj = Projector()
    proj.load_state_dict(sd)
    return proj


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
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        out = self.net(v).view(v.size(0), self.k, -1)
        return self.ln(out)


class ForeignScale(nn.Module):
    def __init__(self, d: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.ln = nn.LayerNorm(d, elementwise_affine=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = F.normalize(v, p=2, dim=-1)
        v = self.ln(v)
        return v * self.scale


class BOWHead(nn.Module):
    def __init__(self, d: int, vocab_size: int) -> None:
        super().__init__()
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pooled = slots.mean(dim=1)
        return self.head(pooled)


class IdentityGatedAdapter(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int = 2048, g0: float = 0.5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.g = nn.Parameter(torch.tensor(g0))
        nn.init.xavier_uniform_(self.w1.weight, gain=0.5)
        nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.5)
        nn.init.zeros_(self.w2.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.w2(F.gelu(self.w1(self.ln(x))))
        gate = torch.tanh(self.g).clamp(0, 0.7)
        return x + gate * residual


def restore_special_rows(model: nn.Module, tokenizer, path: str) -> None:
    if not os.path.exists(path):
        return
    blob = torch.load(path, map_location="cpu")
    token_ids = blob.get("tok_ids", {})
    embed_weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for key, idx in token_ids.items():
            row = torch.tensor(
                blob["input_rows"][key],
                dtype=embed_weight.dtype,
                device=embed_weight.device,
            )
            embed_weight[idx].copy_(row)


def model_dtype(model: nn.Module) -> torch.dtype:
    for param in model.parameters():
        return param.dtype
    return torch.float32


def safe_info_nce(p_f: torch.Tensor, h_e: torch.Tensor, tau: float) -> torch.Tensor:
    p = F.normalize(p_f.float(), dim=-1)
    h = F.normalize(h_e.detach().float(), dim=-1)
    if not torch.isfinite(p).all() or not torch.isfinite(h).all():
        return torch.zeros((), device=p.device)

    sim = p @ h.t()
    if not torch.isfinite(sim).all():
        return torch.zeros((), device=p.device)

    tau = max(float(tau), 1e-2)
    logits = (sim / tau).clamp(-32, 32)
    targets = torch.arange(logits.size(0), device=logits.device)
    log_probs = logits.log_softmax(dim=1)
    loss = F.nll_loss(log_probs, targets, reduction="mean")
    if not torch.isfinite(loss):
        return torch.zeros((), device=logits.device)
    return loss


def teacher_inputs_with_emb(
    tokenizer, texts: List[str], emb_tok: str, max_len: int, device: str
):
    emb_id = tokenizer.convert_tokens_to_ids(emb_tok)
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len - 1,
        add_special_tokens=True,
    ).to(device)
    column = torch.full(
        (batch.input_ids.size(0), 1), emb_id, dtype=batch.input_ids.dtype, device=device
    )
    ids = torch.cat([batch.input_ids, column], 1)
    attn = torch.cat([batch.attention_mask, torch.ones_like(column)], 1)
    pos = torch.full(
        (batch.input_ids.size(0),), ids.size(1) - 1, dtype=torch.long, device=device
    )
    return ids, attn, pos
