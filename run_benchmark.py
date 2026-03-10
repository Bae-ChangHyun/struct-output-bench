#!/usr/bin/env python3
"""Structured Output Benchmark CLI.

Usage:
    # 전체 벤치마크 (모든 데이터셋 × 모든 프레임워크)
    uv run python run_benchmark.py --dataset all

    # 특정 데이터셋
    uv run python run_benchmark.py --dataset deepjsoneval
    uv run python run_benchmark.py --dataset extractbench

    # 특정 프레임워크만
    uv run python run_benchmark.py --dataset deepjsoneval --frameworks instructor/tools openai/default

    # 샘플 수 제한 (기본: 전체 데이터셋)
    uv run python run_benchmark.py --dataset deepjsoneval --max-samples 10

    # 이전 실행 이어하기 (완료된 프레임워크 건너뛰기 + 동일 데이터)
    uv run python run_benchmark.py --resume results/deepjsoneval_20260304_163132

    # 서버 설정
    uv run python run_benchmark.py --dataset deepjsoneval --base-url http://localhost:8001/v1 --model my-model
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from app.benchmark.config import ALL_FW_MODES
from app.benchmark.datasets import get_dataset, list_datasets
from app.benchmark.runner import run_benchmark, print_summary, save_results


def parse_fw_modes(fw_strings: list[str] | None) -> list[tuple[str, str]]:
    """'instructor/tools' 형식 → ('instructor', 'tools') 튜플 리스트."""
    if not fw_strings or fw_strings == ["all"]:
        return ALL_FW_MODES
    modes = []
    for s in fw_strings:
        if "/" in s:
            fw, mode = s.split("/", 1)
            modes.append((fw, mode))
        else:
            # 프레임워크명만 주면 해당 프레임워크의 모든 모드
            matched = [(fw, m) for fw, m in ALL_FW_MODES if fw == s]
            if matched:
                modes.extend(matched)
            else:
                logger.warning(f"Unknown framework '{s}', skipping.")
    return modes


def _save_run_config(run_dir: Path, config: dict):
    """run_config.json 저장 (이어하기용)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _load_run_config(run_dir: Path) -> dict:
    """run_config.json 로드."""
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _get_completed_fw_modes(run_dir: Path) -> set[tuple[str, str]]:
    """이미 완료된 fw/mode 조합 확인."""
    completed = set()
    if not run_dir.exists():
        return completed
    for f in run_dir.glob("*.json"):
        if f.name in ("all.json", "run_config.json"):
            continue
        # {fw}--{mode}.json → (fw, mode)
        stem = f.stem
        parts = stem.split("--", 1)
        if len(parts) == 2:
            completed.add((parts[0], parts[1]))
    return completed


async def run_single_dataset(args, dataset_name: str, fw_modes: list[tuple[str, str]]):
    """단일 데이터셋 벤치마크 실행."""
    adapter = get_dataset(dataset_name)

    # 데이터셋별 로드 인자
    load_kwargs: dict = {}
    if args.max_samples:
        load_kwargs["max_samples"] = args.max_samples
    if args.seed is not None:
        load_kwargs["seed"] = args.seed
    if dataset_name == "extractbench":
        if args.max_text_length:
            load_kwargs["max_text_length"] = args.max_text_length
    elif dataset_name == "custom":
        load_kwargs["path"] = args.custom_path

    samples = adapter.load_samples(**load_kwargs)
    if not samples:
        logger.warning(f"No samples loaded for {dataset_name}. Skipping.")
        return

    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")

    # 프롬프트 오버라이드
    if args.prompt:
        from app.prompts.loader import load_prompt
        prompt_tpl = load_prompt(args.prompt)
        adapter.minimal_prompt = prompt_tpl.system_prompt
        logger.info(f"Using custom prompt: {args.prompt}")

    # 조합 필터링
    combinations = None
    if args.combos:
        from app.benchmark.config import COMBINATIONS
        combinations = [c for c in COMBINATIONS if c["id"] in args.combos]
        logger.info(f"Running combos: {[c['id'] for c in combinations]}")

    # 출력 디렉토리 사전 결정 (프레임워크별 개별 저장)
    base_dir = Path(args.output_dir) if args.output_dir else Path("results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{dataset_name}_{ts}"

    # run_config.json 저장
    run_config = {
        "dataset": dataset_name,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "max_text_length": getattr(args, "max_text_length", None),
        "sample_ids": [s["id"] for s in samples],
        "model": args.model,
        "base_url": args.base_url,
        "combos": args.combos,
        "timestamp": ts,
    }
    _save_run_config(run_dir, run_config)

    results = await run_benchmark(
        adapter=adapter,
        samples=samples,
        fw_modes=fw_modes,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        combinations=combinations,
        save_predictions=not args.no_save_predictions,
        output_dir=run_dir,
    )

    print_summary(results, fw_modes, combinations)
    save_results(results, dataset_name, output_dir=run_dir)


async def resume_run(args, run_dir: Path, fw_modes: list[tuple[str, str]]):
    """기존 실행 이어하기."""
    config = _load_run_config(run_dir)
    dataset_name = config["dataset"]

    # 사용자가 샘플링 인자를 지정하면 경고
    overridden = []
    if args.max_samples is not None:
        overridden.append(f"--max-samples={args.max_samples}")
    if args.seed is not None:
        overridden.append(f"--seed={args.seed}")
    if getattr(args, "max_text_length", None) is not None:
        overridden.append(f"--max-text-length={args.max_text_length}")
    if overridden:
        logger.warning(
            f"Resume 모드: {', '.join(overridden)} 인자가 무시됩니다. "
            f"이전 실행의 동일 데이터(seed={config.get('seed')}, "
            f"max_samples={config.get('max_samples')})를 사용합니다."
        )

    # 이전과 동일한 샘플 로드
    adapter = get_dataset(dataset_name)
    saved_ids = set(config.get("sample_ids", []))

    load_kwargs: dict = {}
    if config.get("max_samples"):
        load_kwargs["max_samples"] = config["max_samples"]
    if config.get("seed") is not None:
        load_kwargs["seed"] = config["seed"]
    if config.get("max_text_length"):
        load_kwargs["max_text_length"] = config["max_text_length"]

    all_samples = adapter.load_samples(**load_kwargs)

    # sample_ids로 필터링 (동일 데이터 보장)
    if saved_ids:
        samples = [s for s in all_samples if s["id"] in saved_ids]
        if len(samples) != len(saved_ids):
            logger.warning(
                f"일부 샘플 매칭 실패: 저장={len(saved_ids)}, 로드={len(samples)}. "
                f"로드된 샘플로 진행합니다."
            )
    else:
        samples = all_samples

    if not samples:
        logger.error(f"No samples loaded for resume. Aborting.")
        return

    # 완료된 fw_mode 스킵
    completed = _get_completed_fw_modes(run_dir)
    remaining = [(fw, m) for fw, m in fw_modes if (fw, m) not in completed]

    if not remaining:
        logger.info("모든 프레임워크가 이미 완료되었습니다.")
        # 기존 결과 로드 후 summary 출력
        existing = []
        for f in run_dir.glob("*.json"):
            if f.name in ("all.json", "run_config.json"):
                continue
            with open(f) as fp:
                existing.extend(json.load(fp))
        if existing:
            print_summary(existing, fw_modes)
        return

    skipped = [(fw, m) for fw, m in fw_modes if (fw, m) in completed]
    if skipped:
        logger.info(f"건너뛰기 (이미 완료): {[f'{fw}/{m}' for fw, m in skipped]}")
    logger.info(f"실행 대상: {[f'{fw}/{m}' for fw, m in remaining]} ({len(samples)} samples)")

    # 조합 필터링
    combinations = None
    saved_combos = config.get("combos")
    if saved_combos:
        from app.benchmark.config import COMBINATIONS
        combinations = [c for c in COMBINATIONS if c["id"] in saved_combos]

    results = await run_benchmark(
        adapter=adapter,
        samples=samples,
        fw_modes=remaining,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        combinations=combinations,
        save_predictions=not args.no_save_predictions,
        output_dir=run_dir,
    )

    # 기존 결과 + 새 결과 합쳐서 summary
    existing = []
    for f in run_dir.glob("*.json"):
        if f.name in ("all.json", "run_config.json"):
            continue
        with open(f) as fp:
            existing.extend(json.load(fp))
    print_summary(existing, fw_modes, combinations)
    save_results(existing, dataset_name, output_dir=run_dir)


async def async_main(args):
    # --resume 모드
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            logger.error(f"Resume 디렉토리가 존재하지 않습니다: {run_dir}")
            return
        fw_modes = parse_fw_modes(args.frameworks)
        if not fw_modes:
            logger.error("No valid frameworks specified.")
            return
        await resume_run(args, run_dir, fw_modes)
        return

    fw_modes = parse_fw_modes(args.frameworks)
    if not fw_modes:
        logger.error("No valid frameworks specified.")
        return

    if args.dataset == "all":
        for name in list_datasets():
            if name == "custom":
                continue
            await run_single_dataset(args, name, fw_modes)
    else:
        await run_single_dataset(args, args.dataset, fw_modes)


def main():
    parser = argparse.ArgumentParser(
        description="Structured Output Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python run_benchmark.py --dataset deepjsoneval
  uv run python run_benchmark.py --dataset all --frameworks instructor/tools openai/default
  uv run python run_benchmark.py --dataset custom --custom-path data/my.jsonl
  uv run python run_benchmark.py --dataset deepjsoneval --base-url http://localhost:8001/v1

Available datasets: deepjsoneval, extractbench, custom, all
Available frameworks: """ + ", ".join(f"{fw}/{m}" for fw, m in ALL_FW_MODES),
    )

    # 데이터셋 (--resume 시 불필요)
    parser.add_argument(
        "--dataset", "-d",
        choices=list_datasets() + ["all"],
        default=None,
        help="데이터셋 이름 (all: 모든 데이터셋 순차 실행)",
    )

    # 이어하기
    parser.add_argument(
        "--resume", "-r", type=str, default=None,
        help="이전 실행 디렉토리에서 이어하기 (완료된 프레임워크 건너뛰기 + 동일 데이터 사용)",
    )

    # 프레임워크
    parser.add_argument(
        "--frameworks", "-f",
        nargs="+",
        default=["all"],
        help="프레임워크/모드 목록 (예: instructor/tools openai/default). 'all'=전체",
    )

    # 서버 설정
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="vLLM 서버 URL")
    parser.add_argument("--model", default=os.getenv("MODEL"), help="모델명")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="API key")

    # 샘플 설정
    parser.add_argument("--max-samples", type=int, default=None, help="최대 샘플 수 (기본: 전체)")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 샘플링 시드 (재현 시 지정, 기본: 매번 랜덤)")
    parser.add_argument("--max-text-length", type=int, default=None, help="[extractbench] 최대 텍스트 길이 필터")

    # 커스텀 데이터셋
    parser.add_argument("--custom-path", type=str, help="커스텀 JSONL 데이터셋 경로")

    # 프롬프트
    parser.add_argument(
        "--prompt", "-p", type=str, default=None,
        help="minimal 프롬프트 템플릿 이름 (app/prompts/templates/ 내 yaml 파일명, 확장자 제외)",
    )
    parser.add_argument(
        "--combos", "-c", nargs="+", default=None,
        choices=["A_desc", "B_nodesc", "C_rich", "D_both"],
        help="실행할 조합 선택 (기본: 전체). 예: -c A_desc C_rich D_both",
    )

    # 출력
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="결과 저장 디렉토리 (기본: results/)")
    parser.add_argument("--no-save-predictions", action="store_true", help="GT/Predicted 저장 안 함 (용량 절약)")

    args = parser.parse_args()

    # validation
    if not args.resume and not args.dataset:
        parser.error("--dataset 또는 --resume 중 하나는 필수입니다.")
    if args.dataset == "custom" and not args.custom_path:
        parser.error("--dataset custom requires --custom-path")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
