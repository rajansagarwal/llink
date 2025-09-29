import json
import logging
import os
import re
import time
import warnings

from transformers.utils import logging as hf_logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "t": int(time.time()),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "name": record.name,
        }
        for key, value in record.__dict__.items():
            if key in {"args", "msg", "message", "exc_text", "exc_info"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    console_level: str = "INFO", json_path: str = "/vol/logs/train.jsonl"
) -> logging.Logger:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    console_level = os.getenv("CONSOLE_LEVEL", console_level).upper()
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level, logging.INFO))
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(json_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    noisy = [
        "hpack",
        "httpx",
        "httpcore",
        "urllib3",
        "grpc",
        "modal",
        "opentelemetry",
        "asyncio",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    token_pattern = re.compile(r"(x-modal-auth-token[^\w-]*)([\w\.-]+)")

    class _Redact(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if isinstance(record.msg, str):
                record.msg = token_pattern.sub(r"\1[REDACTED]", record.msg)
            return True

    console.addFilter(_Redact())
    file_handler.addFilter(_Redact())

    hf_logging.set_verbosity_warning()
    warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces`.*")

    return logging.getLogger("train")
