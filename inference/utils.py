from __future__ import annotations

import os
import torch


def restore_special_rows(model, tokenizer, path: str) -> None:
    if not path or not os.path.exists(path):
        return
    blob = torch.load(path, map_location="cpu")
    embedding_weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for key, idx in blob["tok_ids"].items():
            row = torch.tensor(
                blob["input_rows"][key],
                dtype=embedding_weight.dtype,
                device=embedding_weight.device,
            )
            embedding_weight[idx].copy_(row)
