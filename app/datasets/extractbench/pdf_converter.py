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
