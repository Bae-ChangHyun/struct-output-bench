"""DeepJSONEval 데이터셋 로더.

JSONL 파일에서 샘플을 로드하고 category/depth로 필터링한다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from app.datasets.deepjsoneval.downloader import ensure_dataset


def load_samples(
    max_samples: int | None = None,
    categories: list[str] | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """데이터셋 로드 및 필터링.

    Args:
        max_samples: 최대 반환 샘플 수 (None이면 전체)
        categories: 필터링할 카테고리 목록 (None이면 전체)
        min_depth: 최소 depth 필터
        max_depth: 최대 depth 필터

    Returns:
        list of dict: {id, text, schema_dict, ground_truth, category, true_depth}
    """
    jsonl_path = ensure_dataset()
    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            category = row.get("category", "unknown")
            depth = row.get("true_depth", 0)

            # 필터링
            if categories and category not in categories:
                continue
            if min_depth is not None and depth < min_depth:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            schema_raw = row.get("schema", "{}")
            schema_dict = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
            gt_raw = row.get("json", "{}")
            gt_dict = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw

            samples.append({
                "id": f"djeval_{idx:04d}",
                "text": row.get("text", ""),
                "schema_dict": schema_dict,
                "ground_truth": gt_dict,
                "category": category,
                "true_depth": depth,
            })

    if max_samples and len(samples) > max_samples:
        rng = random.Random(seed)  # seed=None이면 매번 랜덤
        samples = rng.sample(samples, max_samples)

    return samples
