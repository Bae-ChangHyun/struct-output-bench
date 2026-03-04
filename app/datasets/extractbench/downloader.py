"""런타임에 GitHub에서 ExtractBench 레포를 shallow clone."""
from __future__ import annotations

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "extractbench"
REPO_URL = "https://github.com/ContextualAI/extract-bench.git"


def ensure_dataset() -> Path:
    """데이터셋이 없으면 git clone. 있으면 경로만 반환."""
    repo_dir = DATA_DIR / "repo"
    dataset_dir = repo_dir / "dataset"
    if dataset_dir.exists():
        return dataset_dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
        check=True,
    )
    return dataset_dir
