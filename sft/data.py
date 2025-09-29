from __future__ import annotations

import json
from typing import Dict, List

from .config import TrainConfig

_ALLOWED_TYPES = {
    "translate_to_english",
    "paraphrase_in_english",
    "summarize_in_english",
    "qa_about_text",
    "extract_key_fields",
    "title_generation",
    "executive_summary",
    "extract_main_points",
}
_MIN_KH_LEN = 12
_EMB_TOKEN = "<foreign_emb>"


def load_jsonl(path: str, limit: int) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def keep_example(row: Dict, cfg: TrainConfig) -> bool:
    task_type = (row.get("task_type") or "").strip()
    if cfg.translate_only:
        if task_type != "translate_to_english":
            return False
    elif task_type not in _ALLOWED_TYPES:
        return False

    prompt = (row.get("input") or "").strip()
    target = (row.get("output") or row.get("response") or "").strip()
    km = (row.get("foreign_raw") or "").strip()
    if not (prompt and target and km):
        return False
    if _EMB_TOKEN not in prompt:
        return False
    if not (_MIN_KH_LEN <= len(km) <= cfg.max_khmer_chars):
        return False
    return True
