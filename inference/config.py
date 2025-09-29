from dataclasses import dataclass


@dataclass
class InferenceConfig:
    llama_repo: str = "meta-llama/Llama-3.2-1B-Instruct"
    xlm: str = "xlm-roberta-base"

    # Artifacts (final stage-B outputs)
    lora_dir: str = "/vol/ckpts/lora_stageB_final"
    adapter_path: str = "/vol/ckpts/prefix_adapter_stageB.pt"
    expander_path: str = "/vol/ckpts/expander_stageB_final_2.pt"
    foreign_scale_path: str = "/vol/ckpts/foreign_scale_final_2.pt"
    projector_path: str = "/vol/ckpts/projector_stageA_contrastive.pt"
    special_rows_path: str = "/vol/ckpts/special_token_rows.pt"

    # Tokens / sequence lengths
    emb_tok: str = "<foreign_emb>"
    open_tok: str = "<foreign>"
    close_tok: str = "</foreign>"
    k_slots: int = 8
    max_khmer_chars: int = 256
    max_src_tokens: int = 128
    max_new_tokens: int = 128


def default_cfg() -> InferenceConfig:
    return InferenceConfig()
