from modal import Image, App, Volume

BASE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
app = App("khmer-bridge-training")

VOL_NAME = "khmer-bridge-vol"
vol = Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    Image.from_registry(BASE, add_python="3.10")
    .apt_install("git", "libaio1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.0",
        "transformers>=4.44.2",
        "accelerate>=0.33.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.4",
        "tqdm>=4.66.4",
        "matplotlib>=3.7.1",
    )
    .add_local_file(local_path="../data/en-km.jsonl", remote_path="/data/en-km.jsonl")
)
