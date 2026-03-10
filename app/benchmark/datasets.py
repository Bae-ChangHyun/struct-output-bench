"""데이터셋 로더 디스패치.

각 데이터셋은 통일된 인터페이스를 제공:
  load_samples(**kwargs) -> list[dict]
  json_schema_to_pydantic(schema, with_descriptions, model_name) -> type[BaseModel]
  generate_rich_prompt(schema) -> str

반환되는 sample dict 필수 키: {id, text, schema_dict (dict), ground_truth (dict)}
"""
from __future__ import annotations

from typing import Any, Callable


class DatasetAdapter:
    """데이터셋 어댑터. 로딩/스키마변환/프롬프트생성 함수를 묶는 래퍼."""

    def __init__(
        self,
        name: str,
        load_fn: Callable,
        schema_fn: Callable,
        prompt_fn: Callable,
        minimal_prompt: str,
        schema_key_fn: Callable | None = None,
    ):
        self.name = name
        self.load_fn = load_fn
        self.schema_fn = schema_fn
        self.prompt_fn = prompt_fn
        self.minimal_prompt = minimal_prompt
        # ExtractBench처럼 스키마 그룹이 있는 경우, sample → schema_key 매핑 함수
        self.schema_key_fn = schema_key_fn

    def load_samples(self, **kwargs) -> list[dict]:
        return self.load_fn(**kwargs)

    def get_schema_dict(self, sample: dict) -> dict:
        return sample.get("schema_dict", {})

    def get_ground_truth(self, sample: dict) -> dict:
        return sample.get("ground_truth", {})


def get_dataset(name: str) -> DatasetAdapter:
    """이름으로 데이터셋 어댑터 로드."""
    if name == "deepjsoneval":
        return _load_deepjsoneval()
    elif name == "extractbench":
        return _load_extractbench()
    elif name == "custom":
        return _load_custom()
    else:
        available = ["deepjsoneval", "extractbench", "custom"]
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")


def list_datasets() -> list[str]:
    return ["deepjsoneval", "extractbench", "custom"]


def _load_deepjsoneval() -> DatasetAdapter:
    from app.datasets.deepjsoneval import (
        load_samples,
        json_schema_to_pydantic,
        generate_rich_prompt,
    )
    from app.benchmark.config import DEFAULT_MINIMAL_PROMPT

    return DatasetAdapter(
        name="deepjsoneval",
        load_fn=load_samples,
        schema_fn=json_schema_to_pydantic,
        prompt_fn=generate_rich_prompt,
        minimal_prompt=DEFAULT_MINIMAL_PROMPT,
    )


def _load_extractbench() -> DatasetAdapter:
    from app.datasets.extractbench.loader import load_samples
    from app.datasets.shared.schema_converter import json_schema_to_pydantic
    from app.datasets.extractbench.prompt_generator import generate_rich_prompt
    from app.benchmark.config import DEFAULT_MINIMAL_PROMPT

    def schema_key(sample: dict) -> str:
        return f"{sample.get('domain', '')}/{sample.get('schema_name', '')}"

    return DatasetAdapter(
        name="extractbench",
        load_fn=load_samples,
        schema_fn=json_schema_to_pydantic,
        prompt_fn=generate_rich_prompt,
        minimal_prompt=DEFAULT_MINIMAL_PROMPT,
        schema_key_fn=schema_key,
    )


def _load_custom() -> DatasetAdapter:
    """커스텀 JSONL 데이터셋 로더.

    JSONL 형식: 각 줄에 {"text": "...", "schema": {...}, "ground_truth": {...}}
    --custom-path 인자로 경로를 지정해야 함.
    """
    from app.datasets.shared.schema_converter import json_schema_to_pydantic
    from app.datasets.deepjsoneval.prompt_generator import generate_rich_prompt
    from app.benchmark.config import DEFAULT_MINIMAL_PROMPT

    def _load_custom_samples(path: str = "", max_samples: int | None = None, seed: int | None = None, **_kwargs) -> list[dict]:
        import json
        import random
        from pathlib import Path

        if not path:
            raise ValueError("Custom dataset requires --custom-path argument")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Custom dataset not found: {path}")

        samples = []
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                schema = row.get("schema", {})
                if isinstance(schema, str):
                    schema = json.loads(schema)
                gt = row.get("ground_truth", row.get("json", {}))
                if isinstance(gt, str):
                    gt = json.loads(gt)

                samples.append({
                    "id": row.get("id", f"custom_{idx:04d}"),
                    "text": row.get("text", ""),
                    "schema_dict": schema,
                    "ground_truth": gt,
                })

        if max_samples and len(samples) > max_samples:
            rng = random.Random(seed)
            samples = rng.sample(samples, max_samples)

        return samples

    return DatasetAdapter(
        name="custom",
        load_fn=_load_custom_samples,
        schema_fn=json_schema_to_pydantic,
        prompt_fn=generate_rich_prompt,
        minimal_prompt=DEFAULT_MINIMAL_PROMPT,
    )
