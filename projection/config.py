from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # data config
    max_pairs: int = 100_000
    dev_pairs: int = 10_000
    max_khmer_chars: int = 256
    max_src_tokens: int = 128

    # training config
    micro_bsz: int = 8
    grad_accum: int = 16
    total_steps: int = 20_000
    warmup_steps: int = 3_000
    base_lr: float = 1e-3
    min_lr: float = 1e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.05

    # contrastive, regularization
    tau_start: float = 0.17
    tau_end: float = 0.08
    align_cos_w: float = 0.05
    lognorm_w: float = 0.02

    # compute
    use_bf16: bool = True
    allow_tf32: bool = True
    teacher_chunk_bsz: int = 4

    # models
    xlm: str = "xlm-roberta-base"
    llama_repo: str = "meta-llama/Llama-3.2-1B-Instruct"


def default_cfg() -> TrainingConfig:
    return TrainingConfig()
