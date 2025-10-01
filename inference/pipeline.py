from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .config import InferenceConfig
from .models import (
    ForeignScale,
    IdentityGatedAdapter,
    TokenExpander,
    load_projector_from_state,
)
from .utils import restore_special_rows


@dataclass
class StageBComponents:
    tokenizer: AutoTokenizer
    base_model: AutoModelForCausalLM
    x_tokenizer: AutoTokenizer
    x_encoder: AutoModel
    projector: torch.nn.Module
    adapter: IdentityGatedAdapter
    expander: TokenExpander
    foreign_scale: ForeignScale
    device: str


def load_stage_b_components(
    cfg: InferenceConfig, hf_token: str, device: str
) -> StageBComponents:
    x_tokenizer = AutoTokenizer.from_pretrained(cfg.xlm, use_fast=True)
    x_encoder = AutoModel.from_pretrained(cfg.xlm).to(device).eval()

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.llama_repo,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llama_repo, use_fast=True, token=hf_token
    )

    special_tokens = [cfg.open_tok, cfg.close_tok, cfg.emb_tok] + [
        f"<f{i}>" for i in range(cfg.k_slots)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))

    restore_special_rows(base_model, tokenizer, cfg.special_rows_path)

    if not os.path.isdir(cfg.lora_dir):
        raise FileNotFoundError(f"LoRA weights not found at {cfg.lora_dir}")
    base_model = PeftModel.from_pretrained(base_model, cfg.lora_dir)
    base_model.eval()

    projector_state = torch.load(cfg.projector_path, map_location="cpu")
    projector = load_projector_from_state(projector_state).to(device).eval()

    hidden = base_model.get_input_embeddings().weight.size(1)

    adapter = IdentityGatedAdapter(dim=hidden, hidden=hidden).to(device).eval()
    adapter.load_state_dict(torch.load(cfg.adapter_path, map_location="cpu"))

    expander = TokenExpander(d=hidden, k=cfg.k_slots).to(device).eval()
    expander.load_state_dict(torch.load(cfg.expander_path, map_location="cpu"))

    foreign_scale = ForeignScale(hidden).to(device).eval()
    if os.path.exists(cfg.foreign_scale_path):
        foreign_scale.load_state_dict(
            torch.load(cfg.foreign_scale_path, map_location="cpu")
        )
    else:
        with torch.no_grad():
            median = base_model.get_input_embeddings().weight.norm(dim=1).median()
            foreign_scale.scale.data.fill_(float(median))

    return StageBComponents(
        tokenizer=tokenizer,
        base_model=base_model,
        x_tokenizer=x_tokenizer,
        x_encoder=x_encoder,
        projector=projector,
        adapter=adapter,
        expander=expander,
        foreign_scale=foreign_scale,
        device=device,
    )


def build_prompt(cfg: InferenceConfig, task_type: str, strict: bool) -> str:
    if task_type == "translate_to_english":
        instruction = "English translation (one short sentence):"
    elif task_type in {"summarize_in_english", "executive_summary"}:
        instruction = "Summarize the foreign text in English."
    elif task_type == "qa_about_text":
        instruction = "Answer questions about the foreign text in English."
    elif task_type == "title_generation":
        instruction = "Generate a title for the foreign text in English."
    elif task_type == "extract_main_points":
        instruction = "Extract the main points from the foreign text in English."
    else:
        instruction = (
            f"{task_type.replace('_', ' ').title()} the foreign text in English."
        )

    suffix = "Answer in English only."
    if strict:
        suffix += (
            " Output ONLY the translation; do not explain or describe. No extra words."
        )

    slot_tokens = " ".join(f"<f{i}>" for i in range(cfg.k_slots))
    user_input = (
        f"Instruction: {instruction}\n"
        f"Foreign: {cfg.open_tok}{slot_tokens}{cfg.close_tok}\n"
        f"{suffix}"
    )
    return f"User: {user_input}\nAssistant:"


def tokenize_prompt(
    cfg: InferenceConfig, tokenizer, prompt: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_src_tokens,
        add_special_tokens=True,
    )
    input_ids = encoded.input_ids.to(device)
    attention = encoded.attention_mask.to(device)

    slot_ids = [tokenizer.convert_tokens_to_ids(f"<f{i}>") for i in range(cfg.k_slots)]
    positions: List[int] = []
    for idx, slot_id in enumerate(slot_ids):
        match = (input_ids == slot_id).nonzero(as_tuple=False)
        if match.numel() == 0:
            raise ValueError(f"<f{idx}> token not found in prompt")
        positions.append(int(match[0, 1].item()))
    return input_ids, attention, positions


def stage_a_slots(
    components: StageBComponents,
    cfg: InferenceConfig,
    foreign_text: str,
    gate_boost: float = 1.0,
) -> Dict[str, torch.Tensor]:
    device = components.device
    km = [foreign_text[: cfg.max_khmer_chars]]
    tokenized = components.x_tokenizer(
        km,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    ).to(device)

    with torch.no_grad():
        hidden = components.x_encoder(**tokenized).last_hidden_state
        mask = tokenized.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        projector_outputs = components.projector(pooled)
        scaled = components.foreign_scale(projector_outputs)
        adapted = components.adapter(scaled)
        if abs(gate_boost - 1.0) > 1e-6:
            boosted = scaled + gate_boost * (adapted - scaled)
            target_norm = scaled.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            adapted = boosted * (
                target_norm / boosted.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            )
        slots = components.expander(adapted)

    return {
        "projector": projector_outputs,
        "scaled": scaled,
        "adapted": adapted,
        "slots": slots,
    }


def inject_slots(
    base_model,
    input_ids: torch.Tensor,
    slots: torch.Tensor,
    positions: Sequence[int],
) -> Tuple[torch.Tensor, List[float], List[float]]:
    embeddings = base_model.get_input_embeddings()(input_ids)
    original_norms: List[float] = []
    for slot_index, position in enumerate(positions):
        original = embeddings[0, position, :].clone()
        original_norms.append(float(original.norm()))
        embeddings[0, position, :] = slots[0, slot_index, :].to(embeddings.dtype)
    new_norms = [float(embeddings[0, pos, :].norm()) for pos in positions]
    return embeddings, original_norms, new_norms


def zeroed_embeddings(
    base_model, input_ids: torch.Tensor, positions: Sequence[int]
) -> torch.Tensor:
    embeddings = base_model.get_input_embeddings()(input_ids)
    for position in positions:
        original = base_model.get_input_embeddings()(
            input_ids[:, position : position + 1]
        )[0, 0, :]
        embeddings[0, position, :] = original
    return embeddings
