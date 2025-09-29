import os
import sys

if __package__ is None or __package__ == "":
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_DIR not in sys.path:
        sys.path.append(PROJECT_DIR)
    from modal_setup import Secret, app, image, vol  # type: ignore
    from trainer import run_training  # type: ignore
else:
    from .modal_setup import Secret, app, image, vol
    from .trainer import run_training


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,
    secrets=[Secret.from_name("hf-token")],
    volumes={"/vol": vol},
)
def train_projection_model():
    run_training()


if __name__ == "__main__":
    train_projection_model.remote()
