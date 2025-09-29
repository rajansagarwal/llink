from __future__ import annotations

import os
from typing import List

import torch
import torch.nn.functional as F
from modal import App, Image, Secret, Volume
from transformers import AutoModelForCausalLM

from .config import default_cfg
from .pipeline import (
    build_prompt,
    inject_slots,
    load_stage_b_components,
    stage_a_slots,
    tokenize_prompt,
    zeroed_embeddings,
)

BASE_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
APP_NAME = "khmer-bridge-infer"
VOLUME_NAME = "khmer-bridge-vol"

app = App(APP_NAME)
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    Image.from_registry(BASE_IMAGE, add_python="3.10")
    .apt_install("git", "libaio1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.2",
        "peft>=0.10.0",
        "sentencepiece==0.2.0",
    )
)


def _prepare_environment() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def _baseline_model(hf_token: str, tokenizer, cfg, device: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.llama_repo,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()
    return base_model


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[Secret.from_name("hf-token")])
def infer(foreign_text: str, task_type: str = "translate_to_english", strict: bool = False, gate_boost: float = 1.0):
    cfg = default_cfg()
    device = "cuda"
    hf_token = _prepare_environment()


    components = load_stage_b_components(cfg, hf_token, device)

    prompt = build_prompt(cfg, task_type, strict)

    try:
        input_ids, attention, positions = tokenize_prompt(cfg, components.tokenizer, prompt, device)
    except ValueError as exc:
        return f"ERROR: {exc}"

    slot_outputs = stage_a_slots(components, cfg, foreign_text, gate_boost)

    embeddings, original_norms, new_norms = inject_slots(
        components.base_model, input_ids, slot_outputs["slots"], positions
    )

    position_ids = (attention.cumsum(-1) - 1).clamp_min(0)
    ablation_embeddings = zeroed_embeddings(components.base_model, input_ids, positions)


    bad_words_ids = []
    generation_kwargs = dict(
        inputs_embeds=embeddings,
        attention_mask=attention,
        position_ids=position_ids,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=components.tokenizer.eos_token_id,
        eos_token_id=components.tokenizer.eos_token_id,
        bad_words_ids=bad_words_ids,
        no_repeat_ngram_size=3,
    )

    with torch.no_grad():
        generated_with = components.base_model.generate(**generation_kwargs)
        generation_kwargs["inputs_embeds"] = ablation_embeddings
        generated_zero = components.base_model.generate(**generation_kwargs)

        baseline_prompt = f"User: Translate this to English: {foreign_text}\nAssistant:"
        baseline_inputs = components.tokenizer(
            baseline_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_src_tokens,
            add_special_tokens=True,
        ).to(device)
        baseline_model = _baseline_model(hf_token, components.tokenizer, cfg, device)
        generated_base = baseline_model.generate(
            input_ids=baseline_inputs.input_ids,
            attention_mask=baseline_inputs.attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=components.tokenizer.eos_token_id,
            eos_token_id=components.tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,
            no_repeat_ngram_size=3,
        )

    with_vec = components.tokenizer.decode(generated_with[0], skip_special_tokens=True).strip()
    zero_vec = components.tokenizer.decode(generated_zero[0], skip_special_tokens=True).strip()
    base_vec = components.tokenizer.decode(generated_base[0], skip_special_tokens=True).strip()


    if with_vec == zero_vec:
        print("[WARNING] Foreign vector may not be used (matches ablation).")
    if with_vec == base_vec:
        print("[WARNING] Output matches base Llama; no improvement detected.")

    full_text = components.tokenizer.decode(generated_with[0], skip_special_tokens=True)
    result = full_text.split("Assistant:")[-1].strip() if "Assistant:" in full_text else full_text.strip()
    return result


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[Secret.from_name("hf-token")])
def lexeme_probe(foreign_text: str, cand_csv: str = "banana,bananas,eat,eating,day,good,some"):
    cfg = default_cfg()
    device = "cuda"
    hf_token = _prepare_environment()

    components = load_stage_b_components(cfg, hf_token, device)

    prompt = build_prompt(cfg, "translate_to_english", strict=True)
    input_ids, attention, positions = tokenize_prompt(cfg, components.tokenizer, prompt, device)

    slot_outputs = stage_a_slots(components, cfg, foreign_text, gate_boost=1.0)
    slots = slot_outputs["slots"]
    slot_mean = slots.mean(dim=1)[0].float()

    embeddings = components.base_model.get_input_embeddings()(input_ids)
    for slot_index, pos in enumerate(positions):
        embeddings[0, pos, :] = slots[0, slot_index, :].to(embeddings.dtype)

    with torch.no_grad():
        attn = attention
        position_ids = (attn.cumsum(-1) - 1).clamp_min(0)
        output_with = components.base_model(
            inputs_embeds=embeddings,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        logits_with = output_with.logits[0, -1, :].float()

        zero_embeddings = zeroed_embeddings(components.base_model, input_ids, positions)
        output_zero = components.base_model(
            inputs_embeds=zero_embeddings,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        logits_zero = output_zero.logits[0, -1, :].float()

    expansion = torch.nn.functional.normalize(
        components.base_model.get_input_embeddings().weight.detach().float(), dim=-1
    )
    mean_normalized = torch.nn.functional.normalize(slot_mean, dim=-1)
    similarities = expansion @ mean_normalized

    topk = torch.topk(similarities, k=30)
    print("\n[A] Nearest neighbors to slot-mean (top-30):")
    for rank, (idx, score) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
        print(f"{rank:2d}. {components.tokenizer.decode([idx])!r:20s}  cos={score:.4f}  id={idx}")

    def _forms(phrase: str) -> List[str]:
        variants = {phrase, phrase.lower(), phrase.capitalize()}
        variants.update({" " + v for v in list(variants)})
        return list(variants)

    candidates = [token.strip() for token in cand_csv.split(",") if token.strip()]
    candidate_map = {}
    for candidate in candidates:
        ids = []
        for variant in _forms(candidate):
            encoded = components.tokenizer(variant, add_special_tokens=False).input_ids
            if len(encoded) == 1:
                ids.append(encoded[0])
        candidate_map[candidate] = sorted(set(ids))

    ordering = torch.argsort(-similarities).tolist()
    ranks = {idx: ordering.index(idx) + 1 for idx in range(len(components.tokenizer))}

    print("\n[A'] Candidate token ranks/cosines among neighbors:")
    for candidate, id_list in candidate_map.items():
        row = []
        for idx in id_list:
            row.append(
                f"id={idx} rank={ranks[idx]} cos={similarities[idx].item():.4f} str={components.tokenizer.decode([idx])!r}"
            )
        print(f"  {candidate:10s} -> " + (" | ".join(row) if row else "(no single-token form)"))

    delta = logits_with - logits_zero
    print("\n[B] Candidate next-token logits (WITH vs ZEROED):")
    for candidate, id_list in candidate_map.items():
        row = []
        for idx in id_list:
            row.append(
                f"id={idx} Δ={delta[idx].item():+.3f} with={logits_with[idx].item():.3f} zero={logits_zero[idx].item():.3f} str={components.tokenizer.decode([idx])!r}"
            )
        print(f"  {candidate:10s} -> " + (" | ".join(row) if row else "(no single-token form)"))

    with torch.no_grad():
        top_with = torch.topk(logits_with, k=30).indices.tolist()
        top_zero = torch.topk(logits_zero, k=30).indices.tolist()

    print("\n[C] Next-token top-30 (WITH injection):")
    print([components.tokenizer.decode([idx]) for idx in top_with])
    print("\n[C] Next-token top-30 (ZEROED):")
    print([components.tokenizer.decode([idx]) for idx in top_zero])

    return "OK"


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[Secret.from_name("hf-token")])
def test_batch(foreign_texts: List[str], task_type: str = "translate_to_english"):
    cfg = default_cfg()
    device = "cuda"
    hf_token = _prepare_environment()

    components = load_stage_b_components(cfg, hf_token, device)

    with torch.no_grad():
        km_batch = [text[: cfg.max_khmer_chars] for text in foreign_texts]
        tokenized = components.x_tokenizer(
            km_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        ).to(device)
        hidden = components.x_encoder(**tokenized).last_hidden_state
        mask = tokenized.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        vectors = components.projector(pooled)

        if len(foreign_texts) >= 2:
            cos = F.cosine_similarity(vectors.unsqueeze(0), vectors.unsqueeze(1), dim=-1)
            upper = cos[torch.triu(torch.ones_like(cos), diagonal=1) == 1]
            print(f"[diversity] mean={float(upper.mean()):.3f} max={float(upper.max()):.3f}")
            print(f"[diversity] norms={[float(vectors[i].norm()) for i in range(len(foreign_texts))]}")
            return float(upper.mean()), float(upper.max())

    return None, None


if __name__ == "__main__":
    pass
