"""PDF를 텍스트로 변환하고 결과를 로컬 캐시."""
from __future__ import annotations

from pathlib import Path

import pymupdf

TEXTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "extractbench" / "texts"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """단일 PDF에서 텍스트 추출. pymupdf 사용."""
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def ensure_texts(dataset_dir: Path) -> dict[str, str]:
    """dataset/ 내 모든 PDF를 텍스트로 변환하고 캐시. {relative_stem: text}."""
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    for pdf_path in sorted(dataset_dir.rglob("*.pdf")):
        # _extra 폴더는 gold가 없으므로 건너뜀
        if "_extra" in pdf_path.parts:
            continue
        # 캐시 키: domain/schema/filename_stem
        rel = pdf_path.relative_to(dataset_dir)
        parts = rel.parts  # e.g. ('finance', '10kq', 'pdf+gold', 'adp_10q.pdf')
        domain = parts[0]
        schema_name = parts[1]
        stem = pdf_path.stem
        cache_key = f"{domain}__{schema_name}__{stem}"

        cache_path = TEXTS_DIR / f"{cache_key}.txt"
        if cache_path.exists():
            results[cache_key] = cache_path.read_text(encoding="utf-8")
        else:
            text = extract_text_from_pdf(pdf_path)
            cache_path.write_text(text, encoding="utf-8")
            results[cache_key] = text

    return results
