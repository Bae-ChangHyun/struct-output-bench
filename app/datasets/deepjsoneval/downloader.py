"""런타임에 HuggingFace에서 DeepJSONEval 데이터셋을 다운로드."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from loguru import logger

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "deepjsoneval"

HF_JSONL_URL = (
    "https://huggingface.co/datasets/GTSAIInfraLabSOTAS/DeepJSONEval"
    "/resolve/main/DeepJSONEval.jsonl"
)


def ensure_dataset() -> Path:
    """데이터셋이 없으면 HuggingFace에서 다운로드. 있으면 경로만 반환."""
    jsonl_path = DATA_DIR / "dataset.jsonl"
    if jsonl_path.exists():
        return jsonl_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading DeepJSONEval dataset to {jsonl_path} ...")

    with urlopen(HF_JSONL_URL, timeout=60) as resp:
        raw = resp.read()

    # 원본이 JSONL인지 확인 후 저장
    lines = raw.decode("utf-8").strip().splitlines()
    valid = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # JSON 유효성 검증
            json.loads(line)
            f.write(line + "\n")
            valid += 1

    logger.info(f"Downloaded {valid} samples to {jsonl_path}")
    return jsonl_path
