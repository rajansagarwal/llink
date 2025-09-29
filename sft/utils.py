from __future__ import annotations

import logging
import random
from collections import deque
from typing import Deque

import numpy as np
import torch


def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Rolling:
    def __init__(self, window: int = 100) -> None:
        self._buffer: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._buffer.append(float(value))

    def mean(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)


def log_versions() -> None:
    import numpy
    import torch
    import transformers

    logging.getLogger("train").info(
        "versions",
        extra={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
        },
    )
