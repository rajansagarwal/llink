from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import TrainConfig
from .data import keep_example, load_jsonl
from .logging_utils import setup_logging
from .modeling import (
    BOWHead,
    ForeignScale,
    IdentityGatedAdapter,
    TokenExpander,
    load_projector_from_state,
    model_dtype,
    restore_special_rows,
    safe_info_nce,
    teacher_inputs_with_emb,
)
from .tokenization import tokenize_with_labels_multitoken
from .utils import Rolling, log_versions, set_seed

_STOP_WORDS = [ "the", "a", "an", "to", "of", "in", "and", "or", "is", "are", 
               "it", "this", "that", "for", "on", "with", "as", "by", "be", 
               "at", ".", ",", ":", ";", "!", "?", "'", '"', "(", ")", ]

def _prepare_stop_ids(tokenizer) -> Tuple[int, ...]:
    ids: List[int] = []
    encoded = tokenizer(_STOP_WORDS, add_special_tokens=False).input_ids
    for item in encoded:
        if isinstance(item, list):
            ids.extend(item)
        else:
            ids.append(item)
    return tuple(sorted(set(ids)))


def _build_bow_targets(
    tokenizer, texts: List[str], stop_ids: Tuple[int, ...], device: str
) -> torch.Tensor:
    vocab_size = len(tokenizer)
    stop_set = set(stop_ids)
    targets = torch.zeros((len(texts), vocab_size), device=device)
    for row, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        keep = [
            token_id
            for token_id in ids
            if token_id not in stop_set and token_id != tokenizer.pad_token_id
        ]
        if keep:
            targets[row, torch.tensor(keep, device=device)] = 1.0
    return targets


def _load_projector(cfg: TrainConfig, device: str) -> torch.nn.Module:
    projector_path = "/vol/ckpts/projector_stageA_contrastive.pt"
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Stage-A projector missing at {projector_path}")
    state = torch.load(projector_path, map_location="cpu")
    projector = load_projector_from_state(state).to(device)
    for param in projector.parameters():
        param.requires_grad_(cfg.lr_projector > 0.0)
    return projector


def run_stage_b_training(cfg: TrainConfig) -> None:
    override = os.getenv("TRANSLATE_ONLY_MODE", "").strip().lower()
    if override:
        cfg.translate_only = override in {"1", "true", "yes"}

    logger = setup_logging()
    log_versions()
    set_seed(123)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning(
            "CUDA unavailable, falling back to CPU. Training will be extremely slow."
        )

    if cfg.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cfg.use_bf16 and device == "cuda":
        torch.set_float32_matmul_precision("high")

    raw_train = (
        load_jsonl("/data/sft.jsonl", cfg.max_train * 2)
        if os.path.exists("/data/sft.jsonl")
        else []
    )
    raw_dev = (
        load_jsonl("/data/sft_val.jsonl", cfg.max_dev * 2)
        if os.path.exists("/data/sft_val.jsonl")
        else []
    )
    train = [row for row in raw_train if keep_example(row, cfg)][: cfg.max_train]
    dev = [row for row in raw_dev if keep_example(row, cfg)][: cfg.max_dev]
    logger.info("data loaded", extra={"train_size": len(train), "dev_size": len(dev)})

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    x_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    x_encoder = AutoModel.from_pretrained("xlm-roberta-base").to(device).eval()
    for param in x_encoder.parameters():
        param.requires_grad_(False)

    llama_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", use_fast=True, token=hf_token
    )
    specials = ["<foreign>", "</foreign>", "<foreign_emb>"] + [
        f"<f{i}>" for i in range(cfg.k_slots)
    ]
    added = llama_tokenizer.add_special_tokens({"additional_special_tokens": specials})
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        token=hf_token,
    )
    if added > 0:
        base_model.resize_token_embeddings(len(llama_tokenizer))
    restore_special_rows(
        base_model, llama_tokenizer, "/vol/ckpts/special_token_rows.pt"
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_cfg)
    base_model.train()

    teacher = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        token=hf_token,
    ).eval()
    teacher.resize_token_embeddings(len(llama_tokenizer))
    restore_special_rows(teacher, llama_tokenizer, "/vol/ckpts/special_token_rows.pt")

    projector = _load_projector(cfg, device)
    hidden_size = base_model.get_input_embeddings().weight.size(1)

    adapter = IdentityGatedAdapter(
        dim=hidden_size, hidden=hidden_size, g0=cfg.gate_init
    ).to(device)
    expander = TokenExpander(d=hidden_size, k=cfg.k_slots).to(device)
    foreign_scale = ForeignScale(hidden_size, init_scale=1.0).to(device)
    bow_head = BOWHead(hidden_size, len(llama_tokenizer)).to(device)

    adapter.train()
    expander.train()
    foreign_scale.train()
    bow_head.train()

    params = []
    lora_params = [param for param in base_model.parameters() if param.requires_grad]
    if lora_params:
        params.append(
            {"params": lora_params, "lr": 1e-4, "weight_decay": cfg.weight_decay}
        )

    if cfg.lr_projector > 0.0:
        params.append(
            {
                "params": [p for p in projector.parameters() if p.requires_grad],
                "lr": cfg.lr_projector,
                "weight_decay": 0.0,
            }
        )

    params.append(
        {
            "params": adapter.parameters(),
            "lr": cfg.lr_adapter,
            "weight_decay": cfg.weight_decay,
        }
    )
    params.append(
        {
            "params": expander.parameters(),
            "lr": cfg.lr_adapter,
            "weight_decay": cfg.weight_decay,
        }
    )
    params.append({"params": bow_head.parameters(), "lr": 1e-4, "weight_decay": 0.0})
    params.append(
        {"params": foreign_scale.parameters(), "lr": 5e-4, "weight_decay": 0.0}
    )

    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = (
        math.ceil(len(train) / (cfg.micro_bsz * cfg.grad_accum)) if train else 0
    )
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_frac * total_steps)
    scheduler = (
        get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        if total_steps > 0
        else None
    )

    stop_ids = _prepare_stop_ids(llama_tokenizer)
    rolling = Rolling(100)

    logger.info(
        "model setup",
        extra={
            "base_dtype": str(model_dtype(base_model)),
            "teacher_dtype": str(model_dtype(teacher)),
            "hidden": hidden_size,
            "k_slots": cfg.k_slots,
            "trainable_params": sum(
                p.numel() for group in params for p in group["params"]
            ),
        },
    )

    def inject_and_forward(batch: List[Dict], step: int):
        ids_batch, label_batch, positions_batch = [], [], []
        km_batch, targets = [], []
        references = []
        for row in batch:
            task_type = row.get("task_type", "translate_to_english")
            if task_type == "translate_to_english":
                instruction = "Translate the foreign text to English."
                suffix = "Answer in English only."
            elif task_type == "paraphrase_in_english":
                instruction = "Paraphrase the foreign text in English."
                suffix = "Answer in English only."
            elif task_type in {"summarize_in_english", "executive_summary"}:
                instruction = "Summarize the foreign text in English."
                suffix = "Answer in English only."
            elif task_type == "qa_about_text":
                instruction = "Answer questions about the foreign text in English."
                suffix = "Answer in English only."
            elif task_type == "title_generation":
                instruction = "Generate a title for the foreign text in English."
                suffix = "Answer in English only."
            elif task_type == "extract_main_points":
                instruction = (
                    "Extract the main points from the foreign text in English."
                )
                suffix = "Answer in English only."
            else:
                instruction = f"{task_type.replace('_', ' ').title()} the foreign text in English."
                suffix = "Answer in English only."

            prompt = f"Instruction: {instruction}\nForeign: <foreign><foreign_emb></foreign>\n{suffix}"
            target = (row.get("output") or row.get("response") or "").strip()
            ids, labels, positions = tokenize_with_labels_multitoken(
                llama_tokenizer,
                prompt,
                target,
                cfg.seq_len,
                cfg.k_slots,
                device,
            )
            ids_batch.append(ids)
            label_batch.append(labels)
            positions_batch.append(positions)
            km_batch.append((row.get("foreign_raw") or "")[: cfg.max_khmer_chars])
            references.append((row.get("english_ref") or target).strip())
            targets.append(target)

        max_len = max(t.size(0) for t in ids_batch)
        pad_id = llama_tokenizer.pad_token_id
        batch_size = len(batch)
        input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long, device=device
        )
        labels = torch.full(
            (batch_size, max_len), -100, dtype=torch.long, device=device
        )
        for idx, (ids_tensor, label_tensor) in enumerate(zip(ids_batch, label_batch)):
            input_ids[idx, : ids_tensor.size(0)] = ids_tensor
            labels[idx, : label_tensor.size(0)] = label_tensor

        with torch.no_grad():
            km_encoded = x_tokenizer(
                km_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
            ).to(device)
            pooled = x_encoder(**km_encoded).last_hidden_state
            mask = km_encoded.attention_mask.unsqueeze(-1).float()
            mean = (pooled * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            projector_out = projector(mean)

        scaled = foreign_scale(projector_out)
        adapted = adapter(scaled)
        slots = expander(adapted)

        bow_logits = bow_head(slots)
        bow_targets = _build_bow_targets(llama_tokenizer, targets, stop_ids, device)
        bow_loss = F.binary_cross_entropy_with_logits(
            bow_logits, bow_targets, reduction="mean"
        )

        embeddings = base_model.get_input_embeddings()(input_ids)
        for row_idx, positions in enumerate(positions_batch):
            for slot_idx, pos in enumerate(positions[: cfg.k_slots]):
                if pos < embeddings.size(1):
                    embeddings[row_idx, pos, :] = slots[row_idx, slot_idx, :].to(
                        embeddings.dtype
                    )

        attn = (input_ids != pad_id).long()
        position_ids = (attn.cumsum(-1) - 1).clamp_min(0)

        outputs = base_model(
            inputs_embeds=embeddings,
            attention_mask=attn,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
        )
        nll = outputs.loss.float()

        if step % 3 == 0:
            with torch.no_grad():
                zero_embeddings = embeddings.clone()
                base_embed = base_model.get_input_embeddings()
                for row_idx, positions in enumerate(positions_batch):
                    for slot_idx, pos in enumerate(positions[: cfg.k_slots]):
                        if pos < zero_embeddings.size(1):
                            zero_embeddings[row_idx, pos, :] = base_embed(
                                input_ids[row_idx : row_idx + 1, pos : pos + 1]
                            )[0, 0, :]
            zero_out = base_model(
                inputs_embeds=zero_embeddings,
                attention_mask=attn,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )
            contrast = 0.05 * torch.relu(zero_out.loss.float() - nll)
            nll = nll + contrast

        slot_means = slots.mean(dim=1)
        with torch.no_grad():
            teacher_ids, teacher_attn, teacher_pos = teacher_inputs_with_emb(
                llama_tokenizer,
                references,
                "<foreign_emb>",
                cfg.max_src_tokens,
                device,
            )
            teacher_hidden = teacher.model(
                input_ids=teacher_ids,
                attention_mask=teacher_attn,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]
            teacher_vec = teacher_hidden[
                torch.arange(batch_size, device=device), teacher_pos
            ]

        reg_cos = F.mse_loss(
            F.normalize(slot_means.float(), dim=-1),
            F.normalize(teacher_vec.float(), dim=-1),
        )
        reg_nce = safe_info_nce(
            slot_means.float(), teacher_vec.float(), cfg.info_nce_tau
        )

        return nll, reg_cos.float(), reg_nce.float(), bow_loss.float()

    baseline_ce = None
    if len(dev) >= 5:
        adapter.eval()
        with torch.no_grad():
            nll, _, _, _ = inject_and_forward(dev[: cfg.micro_bsz], 0)
            baseline_ce = float(nll.detach().cpu())
        adapter.train()
        logger.info("baseline", extra={"ce_loss": baseline_ce})

    if total_steps == 0:
        logger.warning("No training data available.")
        return

    base_model.train()
    teacher.eval()
    projector.eval()

    ptr = 0
    best_dev = float("inf")

    with logging_redirect_tqdm():
        loop = trange(
            total_steps, desc="Stage-B", dynamic_ncols=True, miniters=1, smoothing=0.1
        )
        for step in loop:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for _ in range(cfg.grad_accum):
                batch = train[ptr : ptr + cfg.micro_bsz]
                if len(batch) < cfg.micro_bsz:
                    batch = batch + train[: cfg.micro_bsz - len(batch)]
                ptr = (ptr + cfg.micro_bsz) % len(train)

                nll, reg_cos, reg_nce, bow_loss = inject_and_forward(batch, step)
                loss = (
                    nll
                    + cfg.align_cos_w * reg_cos
                    + cfg.info_nce_w * reg_nce
                    + cfg.bow_loss_w * bow_loss
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Encountered non-finite loss")
                (loss / cfg.grad_accum).backward()
                total_loss += float(loss.detach().cpu())

            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters())
                + list(expander.parameters())
                + list(bow_head.parameters())
                + list(foreign_scale.parameters())
                + lora_params,
                cfg.max_grad_norm,
            )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            rolling.add(total_loss / cfg.grad_accum)
            loop.set_postfix_str(
                f"loss={total_loss / cfg.grad_accum:.4f} gate={float(torch.tanh(adapter.g)):.3f}"
            )

            if step % 300 == 0 and dev:
                adapter.eval()
                with torch.no_grad():
                    eval_losses: List[float] = []
                    for idx in range(0, len(dev), cfg.micro_bsz):
                        chunk = dev[idx : idx + cfg.micro_bsz]
                        nll, reg_cos, reg_nce, bow_loss = inject_and_forward(
                            chunk, step
                        )
                        eval_losses.append(
                            float(
                                (
                                    nll
                                    + cfg.align_cos_w * reg_cos
                                    + cfg.info_nce_w * reg_nce
                                    + cfg.bow_loss_w * bow_loss
                                )
                                .detach()
                                .cpu()
                            )
                        )
                    dev_loss = sum(eval_losses) / len(eval_losses)
                    dev_ppl = math.exp(min(dev_loss, 10))
                logger.info(
                    "dev",
                    extra={"step": step, "ce_loss": dev_loss, "perplexity": dev_ppl},
                )
                adapter.train()

                if dev_loss + 1e-6 < best_dev:
                    best_dev = dev_loss
                    os.makedirs("/vol/ckpts", exist_ok=True)
                    base_model.save_pretrained("/vol/ckpts/lora_stageB_best")
                    llama_tokenizer.save_pretrained("/vol/ckpts/lora_stageB_best")
                    torch.save(
                        adapter.state_dict(), "/vol/ckpts/prefix_adapter_stageB_best.pt"
                    )
                    torch.save(
                        expander.state_dict(), "/vol/ckpts/expander_stageB_best_2.pt"
                    )
                    torch.save(
                        bow_head.state_dict(), "/vol/ckpts/bow_head_stageB_best_2.pt"
                    )
                    torch.save(
                        foreign_scale.state_dict(), "/vol/ckpts/foreign_scale_best_2.pt"
                    )
                    if cfg.lr_projector > 0.0:
                        torch.save(
                            projector.state_dict(), "/vol/ckpts/projector_stageB.pt"
                        )

    os.makedirs("/vol/ckpts", exist_ok=True)
    base_model.save_pretrained("/vol/ckpts/lora_stageB_final")
    llama_tokenizer.save_pretrained("/vol/ckpts/lora_stageB_final")
    torch.save(adapter.state_dict(), "/vol/ckpts/prefix_adapter_stageB.pt")
    torch.save(expander.state_dict(), "/vol/ckpts/expander_stageB_final_2.pt")
    torch.save(bow_head.state_dict(), "/vol/ckpts/bow_head_stageB_final_2.pt")
    torch.save(foreign_scale.state_dict(), "/vol/ckpts/foreign_scale_final_2.pt")
    if cfg.lr_projector > 0.0:
        torch.save(projector.state_dict(), "/vol/ckpts/projector_stageB.pt")

    if dev:
        adapter.eval()
        with torch.no_grad():
            losses: List[float] = []
            for idx in range(0, len(dev), cfg.micro_bsz):
                chunk = dev[idx : idx + cfg.micro_bsz]
                nll, reg_cos, reg_nce, bow_loss = inject_and_forward(chunk, total_steps)
                losses.append(
                    float(
                        (
                            nll
                            + cfg.align_cos_w * reg_cos
                            + cfg.info_nce_w * reg_nce
                            + cfg.bow_loss_w * bow_loss
                        )
                        .detach()
                        .cpu()
                    )
                )
            final_dev_loss = sum(losses) / len(losses)
            final_dev_ppl = math.exp(min(final_dev_loss, 10))
        summary = {"final_dev_ce": final_dev_loss, "final_dev_ppl": final_dev_ppl}
        if baseline_ce is not None:
            summary["baseline_ce"] = baseline_ce
            summary["ce_improvement"] = baseline_ce - final_dev_loss
        logger.info("training complete", extra=summary)
