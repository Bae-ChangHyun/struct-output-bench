"""ExtractBench 샘플 로드."""
from __future__ import annotations

import json
from pathlib import Path

from .downloader import ensure_dataset
from .pdf_converter import extract_text_from_pdf, TEXTS_DIR


def load_samples(max_text_length: int = 50000) -> list[dict]:
    """ExtractBench 샘플 로드.

    Args:
        max_text_length: 이 글자 수를 초과하는 문서는 건너뜀 (context window 초과 방지).
                         0이면 필터 없음.

    Returns:
        각 샘플: {id, text, schema_dict, ground_truth, domain, schema_name, pdf_path}
    """
    dataset_dir = ensure_dataset()
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    # dataset/ 내의 domain/schema 디렉토리를 탐색
    for domain_dir in sorted(dataset_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name  # e.g. 'finance', 'academic'

        for schema_dir in sorted(domain_dir.iterdir()):
            if not schema_dir.is_dir():
                continue
            schema_name = schema_dir.name  # e.g. '10kq', 'research'

            # 스키마 파일 찾기: *-schema.json
            schema_files = list(schema_dir.glob("*-schema.json"))
            if not schema_files:
                continue
            schema_path = schema_files[0]
            with open(schema_path) as f:
                schema_dict = json.load(f)

            # pdf+gold 디렉토리 탐색
            pdf_gold_dir = schema_dir / "pdf+gold"
            if not pdf_gold_dir.exists():
                continue

            for pdf_path in sorted(pdf_gold_dir.glob("*.pdf")):
                # _extra 폴더 건너뜀
                if "_extra" in pdf_path.parts:
                    continue

                stem = pdf_path.stem
                gold_path = pdf_gold_dir / f"{stem}.gold.json"
                if not gold_path.exists():
                    continue

                # Gold JSON 로드
                with open(gold_path) as f:
                    ground_truth = json.load(f)

                # PDF -> 텍스트 변환 (캐시)
                cache_key = f"{domain}__{schema_name}__{stem}"
                cache_path = TEXTS_DIR / f"{cache_key}.txt"
                if cache_path.exists():
                    text = cache_path.read_text(encoding="utf-8")
                else:
                    text = extract_text_from_pdf(pdf_path)
                    cache_path.write_text(text, encoding="utf-8")

                if max_text_length and len(text) > max_text_length:
                    continue

                samples.append({
                    "id": cache_key,
                    "text": text,
                    "schema_dict": schema_dict,
                    "ground_truth": ground_truth,
                    "domain": domain,
                    "schema_name": schema_name,
                    "pdf_path": str(pdf_path),
                })

    return samples
