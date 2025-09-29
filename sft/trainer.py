from __future__ import annotations

from modal import App, Image, Secret, Volume

from .config import default_cfg
from .training_loop import run_stage_b_training

BASE_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
APP_NAME = "khmer-bridge-stageB-multi-token"
VOLUME_NAME = "khmer-bridge-vol"

app = App(APP_NAME)
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    Image.from_registry(BASE_IMAGE, add_python="3.10")
    .apt_install("git", "libaio1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "bitsandbytes>=0.43.1",
        "peft>=0.10.0",
        "tqdm==4.66.4",
        "numpy==1.26.4",
        "sentencepiece==0.2.0",
    )
    .add_local_file(local_path="../jsonl/sft.jsonl", remote_path="/data/sft.jsonl")
    .add_local_file(
        local_path="../jsonl/sft_validation.jsonl", remote_path="/data/sft_val.jsonl"
    )
    .add_local_file(local_path="../jsonl/en-km.jsonl", remote_path="/data/en-km.jsonl")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    secrets=[Secret.from_name("hf-token")],
    volumes={"/vol": volume},
)
def train_stage_b() -> None:
    run_stage_b_training(default_cfg())


if __name__ == "__main__":
    train_stage_b.remote()
