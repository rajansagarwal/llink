from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def tokenize_with_labels_multitoken(
    tokenizer,
    prompt: str,
    target: str,
    seq_len: int,
    k_slots: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    slot_tokens = " ".join(f"<f{i}>" for i in range(k_slots))
    prompt_with_slots = prompt.replace("<foreign_emb>", slot_tokens)
    full = f"{prompt_with_slots} {target.strip()}"
    toks = tokenizer(
        full,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
    )
    ids = toks.input_ids[0]
    asst = tokenizer("Assistant:", add_special_tokens=False).input_ids

    def _find(sequence: torch.Tensor, pattern: Sequence[int]) -> int:
        values = sequence.tolist()
        pat = list(pattern)
        for start in range(len(values) - len(pat) + 1):
            if values[start : start + len(pat)] == pat:
                return start
        return -1

    idx = _find(ids, asst)
    labels = ids.clone()
    mask_idx = (idx + len(asst)) if idx >= 0 else ids.size(0) // 2
    labels[:mask_idx] = -100

    slot_ids = [tokenizer.convert_tokens_to_ids(f"<f{i}>") for i in range(k_slots)]
    positions: List[int] = []
    for slot_id in slot_ids:
        pos = (ids == slot_id).nonzero(as_tuple=False)
        if pos.numel() > 0:
            positions.append(int(pos[0].item()))

    if len(positions) < k_slots:
        base = tokenizer(
            prompt_with_slots,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len - k_slots,
            add_special_tokens=True,
        )
        slot_tensor = torch.tensor(slot_ids, dtype=base.input_ids.dtype)
        ids = torch.cat([base.input_ids[0], slot_tensor], 0)
        labels = torch.cat(
            [
                torch.full_like(base.input_ids[0], -100),
                torch.full_like(slot_tensor, -100),
            ],
            0,
        )
        positions = list(
            range(base.input_ids.size(1), base.input_ids.size(1) + k_slots)
        )

    return ids.to(device), labels.to(device), positions
