import json
import math
import os
import random
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .config import TrainingConfig, default_cfg

CHECKPOINT_DIR = "/vol/ckpts"
PROJECTOR_PATH = f"{CHECKPOINT_DIR}/projector_stageA_contrastive.pt"
META_PATH = f"{CHECKPOINT_DIR}/meta_stageA.json"
SPECIAL_ROWS_PATH = f"{CHECKPOINT_DIR}/special_token_rows.pt"
DATA_PATH = "/data/en-km.jsonl"


def cosine_lr(step: int, total: int, warmup: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return base * (step / max(1, warmup))
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (base - min_lr)


class Projector(nn.Module):
    def __init__(self, dropout_prob: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(3072, 2048),
        )
        self.ln = nn.LayerNorm(2048)

    def forward(self, z):
        return self.ln(self.net(z))


def snapshot_special_rows(model, tokenizer, tokens: List[str], path: str) -> None:
    embeddings = model.get_input_embeddings().weight.detach().cpu().float()
    blob = {"input_rows": {}, "tok_ids": {}}
    for tok in tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        blob["tok_ids"][tok] = int(tok_id)
        blob["input_rows"][tok] = embeddings[tok_id].numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)
    norms = {
        k: float(torch.tensor(v).float().norm()) for k, v in blob["input_rows"].items()
    }
    print(f"[snapshot] special rows -> {path} | norms={norms}")


def teacher_inputs_with_emb(
    tokenizer, english: List[str], emb_tok: str, max_src_tokens: int, device
):
    emb_id = tokenizer.convert_tokens_to_ids(emb_tok)
    toks = tokenizer(
        english,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_src_tokens - 1,
        add_special_tokens=True,
    )
    input_ids = toks.input_ids
    attention_mask = toks.attention_mask
    batch_size = input_ids.size(0)
    emb_column = torch.full((batch_size, 1), emb_id, dtype=input_ids.dtype)
    mask_column = torch.ones((batch_size, 1), dtype=attention_mask.dtype)
    input_ids = torch.cat([input_ids, emb_column], 1)
    attention_mask = torch.cat([attention_mask, mask_column], 1)
    emb_pos = torch.full((batch_size,), input_ids.size(1) - 1, dtype=torch.long)
    return input_ids.to(device), attention_mask.to(device), emb_pos.to(device)


@torch.inference_mode()
def teacher_state_at_emb_chunked(
    llama, input_ids, attention_mask, emb_pos, chunk_bsz: int
):
    hidden_size = llama.config.hidden_size
    batch_size = input_ids.size(0)
    outputs = []
    for start in range(0, batch_size, chunk_bsz):
        end = min(batch_size, start + chunk_bsz)
        ids = input_ids[start:end]
        attn = attention_mask[start:end]
        pos = emb_pos[start:end]
        result = llama.model(
            input_ids=ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = result.hidden_states[-1]
        gather_idx = pos.view(-1, 1, 1).expand(-1, 1, hidden_size)
        emb_hidden = hidden_states.gather(1, gather_idx).squeeze(1)
        outputs.append(emb_hidden)
    return torch.cat(outputs, 0)


def xlmr_mean(
    tokenizer, encoder, khmer: List[str], max_chars: int, device
) -> torch.Tensor:
    clipped = [s[:max_chars] for s in khmer]
    tokens = tokenizer(
        clipped,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        hidden_states = encoder(**tokens).last_hidden_state
    mask = tokens.attention_mask.unsqueeze(-1)
    pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1)
    return pooled


def global_metrics(pF, hE):
    p = F.normalize(pF.float(), dim=-1)
    h = F.normalize(hE.float(), dim=-1)
    sim = p @ h.t()
    rank = torch.argsort(-sim, dim=1)
    gold = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    hits = (rank == gold).nonzero(as_tuple=False)[:, 1] + 1
    r1 = (hits <= 1).float().mean().item()
    r5 = (hits <= 5).float().mean().item()
    r10 = (hits <= 10).float().mean().item()
    mrr = (1.0 / hits.float()).mean().item()
    mrk = hits.float().mean().item()
    return dict(r1=r1, r5=r5, r10=r10, mrr=mrr, mean_rank=mrk)


def _eval_on_dev(
    step: int,
    llama,
    l_tok,
    proj,
    x_tok,
    x_enc,
    dev_eval: List[Tuple[str, str]],
    device: str,
    cfg: TrainingConfig,
) -> None:
    with torch.no_grad():
        en = [e for (e, _) in dev_eval]
        km = [k for (_, k) in dev_eval]
        ids, attn, pos = teacher_inputs_with_emb(
            l_tok, en, "<foreign_emb>", cfg.max_src_tokens, device
        )
        hE = teacher_state_at_emb_chunked(llama, ids, attn, pos, cfg.teacher_chunk_bsz)
        pF = proj(xlmr_mean(x_tok, x_enc, km, cfg.max_khmer_chars, device))
        metrics = global_metrics(pF, hE)
        tqdm.write(
            f"[DEV@{step}] R@1={metrics['r1']:.3f} R@5={metrics['r5']:.3f} "
            f"R@10={metrics['r10']:.3f} MRR={metrics['mrr']:.3f} "
            f"MR={metrics['mean_rank']:.1f}"
        )


def run_training(cfg: Optional[TrainingConfig] = None) -> None:
    cfg = cfg or default_cfg()

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.use_bf16:
        torch.set_float32_matmul_precision("high")
    device = "cuda"

    pairs: List[Tuple[str, str]] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            en = (obj.get("english") or "").strip()
            km = (obj.get("foreign") or "").strip()
            if en and km:
                pairs.append((en, km))
    random.shuffle(pairs)
    if cfg.max_pairs is not None:
        pairs = pairs[: cfg.max_pairs + cfg.dev_pairs]
    dev_pairs = pairs[: cfg.dev_pairs]
    train_pairs = pairs[cfg.dev_pairs :]
    print(f"Loaded pairs: train={len(train_pairs)} dev={len(dev_pairs)}")

    hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    if not hf_tok:
        raise AssertionError("Missing HF token")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_tok

    x_tok = AutoTokenizer.from_pretrained(cfg.xlm, use_fast=True)
    x_enc = AutoModel.from_pretrained(cfg.xlm).to(device).eval()
    for param in x_enc.parameters():
        param.requires_grad_(False)

    l_tok = AutoTokenizer.from_pretrained(cfg.llama_repo, use_fast=True, token=hf_tok)
    l_tok.add_special_tokens(
        {"additional_special_tokens": ["<foreign>", "</foreign>", "<foreign_emb>"]}
    )
    if l_tok.pad_token is None:
        l_tok.pad_token = l_tok.eos_token
    l_tok.padding_side = "left"

    llama = AutoModelForCausalLM.from_pretrained(
        cfg.llama_repo,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        token=hf_tok,
    ).eval()
    llama.resize_token_embeddings(len(l_tok))

    snapshot_special_rows(
        llama,
        l_tok,
        ["<foreign>", "</foreign>", "<foreign_emb>"],
        SPECIAL_ROWS_PATH,
    )

    proj = Projector().to(device).train()
    optimizer = torch.optim.AdamW(
        proj.parameters(),
        lr=cfg.base_lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    lr_schedule = lambda step: cosine_lr(
        step, cfg.total_steps, cfg.warmup_steps, cfg.base_lr, cfg.min_lr
    )

    dev_eval_slice = 1024
    # Keep validation slice predictable for controlled VRAM usage during eval
    dev_eval = (
        dev_pairs[:dev_eval_slice] if len(dev_pairs) > dev_eval_slice else dev_pairs
    )

    # Hard-negative queue buffers LLaMA states for contrastive mining
    hidden_size = llama.config.hidden_size
    queue_size = 32768
    hard_negatives = 256
    mem_hE = torch.empty(queue_size, hidden_size, device=device, dtype=torch.float16)
    mem_ptr = 0
    mem_filled = 0

    step = 0
    progress = trange(
        cfg.total_steps, desc="BridgeTrain", dynamic_ncols=True, leave=True
    )
    # Advance through curriculum using micro-batch gradient accumulation
    for _ in progress:
        start = (step * cfg.grad_accum * cfg.micro_bsz) % max(
            1, len(train_pairs) - cfg.grad_accum * cfg.micro_bsz
        )
        chunk = train_pairs[start : start + cfg.grad_accum * cfg.micro_bsz]
        if len(chunk) < cfg.grad_accum * cfg.micro_bsz:
            random.shuffle(train_pairs)
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        tau_frac = min(1.0, step / max(1, cfg.total_steps))
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * tau_frac

        for micro_start in range(0, len(chunk), cfg.micro_bsz):
            micro = chunk[micro_start : micro_start + cfg.micro_bsz]
            # Work on micro-batches to keep chunked teacher forward within memory budget
            en = [e for (e, _) in micro]
            km = [k for (_, k) in micro]
            ids, attn, pos = teacher_inputs_with_emb(
                l_tok, en, "<foreign_emb>", cfg.max_src_tokens, device
            )
            hE = teacher_state_at_emb_chunked(
                llama, ids, attn, pos, cfg.teacher_chunk_bsz
            )
            zF = xlmr_mean(x_tok, x_enc, km, cfg.max_khmer_chars, device)
            pF = proj(zF)

            pN = F.normalize(pF.float(), dim=-1)
            hN = F.normalize(hE.float(), dim=-1)

            if mem_filled > 0:
                mem_slice = mem_hE[:mem_filled]
                simQ = pN @ mem_slice.float().T
                k_eff = min(hard_negatives, mem_filled)
                hard_idx = torch.topk(simQ.mean(0), k_eff).indices
                neg_pool = mem_slice[hard_idx].float()
            else:
                neg_pool = torch.empty(
                    0, hN.size(-1), device=device, dtype=torch.float32
                )

            full_pool = torch.cat([hN, neg_pool], dim=0)
            labels = torch.arange(pN.size(0), device=pN.device)
            logits_i = (pN @ full_pool.T) / tau
            loss_i = F.cross_entropy(logits_i, labels)

            logits_j = (hN @ pN.T) / tau
            loss_j = F.cross_entropy(logits_j, labels)

            ntx_loss = 0.5 * (loss_i + loss_j)

            dir_reg = F.mse_loss(pN, hN)
            log_norm = (pF.float().norm(dim=-1) + 1e-8).log() - (
                hE.float().norm(dim=-1) + 1e-8
            ).log()
            reg_norm = (log_norm**2).mean()
            loss = ntx_loss + cfg.align_cos_w * dir_reg + cfg.lognorm_w * reg_norm

            (loss / cfg.grad_accum).backward()
            total_loss += float(loss.detach().cpu())

            with torch.no_grad():
                # Ring-buffer update: stash normalized teacher states for future negatives
                h_store = F.normalize(hE.detach().to(torch.float16), dim=-1)
                batch_size = h_store.size(0)
                remaining = min(batch_size, queue_size - mem_ptr)
                mem_hE[mem_ptr : mem_ptr + remaining] = h_store[:remaining]
                if remaining < batch_size:
                    mem_hE[0 : batch_size - remaining] = h_store[remaining:]
                mem_ptr = (mem_ptr + batch_size) % queue_size
                mem_filled = min(queue_size, mem_filled + batch_size)

        torch.nn.utils.clip_grad_norm_(proj.parameters(), cfg.max_grad_norm)
        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(step)
        optimizer.step()
        step += 1

        if step % 50 == 0:
            progress.set_postfix_str(
                f"loss={total_loss / cfg.grad_accum:.4f} tau={tau:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if (step + 1) % 500 == 0 or step == 1:
            _eval_on_dev(step, llama, l_tok, proj, x_tok, x_enc, dev_eval, device, cfg)

        if step >= cfg.total_steps:
            break

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(proj.state_dict(), PROJECTOR_PATH)
    # Persist checkpoint metadata so downstream jobs can reload everything
    meta = {
        "llama_repo": cfg.llama_repo,
        "xlm": cfg.xlm,
        "tokens": {
            "open": "<foreign>",
            "close": "</foreign>",
            "emb": "<foreign_emb>",
        },
        "hp": asdict(cfg),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved params -> {PROJECTOR_PATH}")
    print(f"Saved special rows -> {SPECIAL_ROWS_PATH}")
