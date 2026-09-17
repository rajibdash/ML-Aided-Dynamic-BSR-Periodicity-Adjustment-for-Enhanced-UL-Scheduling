from __future__ import annotations

import json
import logging
from pathlib import Path


def project_path(relative_path: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / relative_path


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def dump_json(payload: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
