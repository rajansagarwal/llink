from dataclasses import dataclass


@dataclass
class TrainConfig:
    # data limits
    max_train: int = 40_000
    max_dev: int = 2_000
    max_khmer_chars: int = 256
    max_src_tokens: int = 128
    seq_len: int = 768

    # optimization
    epochs: int = 3
    micro_bsz: int = 8
    grad_accum: int = 16
    lr_adapter: float = 3e-5
    lr_projector: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.15

    # regularizers
    align_cos_w: float = 0.05
    info_nce_w: float = 1e-3
    info_nce_tau: float = 0.07

    # adapters
    gate_init: float = 0.5
    k_slots: int = 8
    bow_loss_w: float = 0.1

    # behaviour
    translate_only: bool = True

    # compute
    use_bf16: bool = True
    tf32: bool = True


def default_cfg() -> TrainConfig:
    return TrainConfig()
