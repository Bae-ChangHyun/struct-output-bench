"""통합 벤치마크 러너.

데이터셋 종류에 관계없이 동일한 로직으로:
  samples × combinations × frameworks → score + save
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.frameworks.registry import FrameworkRegistry
from app.scoring import score_result
from app.benchmark.config import COMBINATIONS, ALL_FW_MODES
from app.benchmark.datasets import DatasetAdapter

import app.frameworks  # noqa: F401  (어댑터 자동 등록)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


async def _run_single(
    fw: str, mode: str, schema_class: type, system_prompt: str, text: str,
    model: str, base_url: str, api_key: str,
) -> dict:
    """단일 프레임워크 추출 실행."""
    adapter_cls = FrameworkRegistry.get(fw)
    adapter = adapter_cls(model=model, base_url=base_url, api_key=api_key, mode=mode)
    result = await adapter.run(
        text=text, schema_class=schema_class, system_prompt=system_prompt,
    )
    return {
        "success": result.success,
        "latency_ms": round(result.latency_ms, 1),
        "data": result.data,
        "error": (result.error or "")[:200] if not result.success else None,
    }


def _prepare_models_and_prompts(
    adapter: DatasetAdapter, samples: list[dict]
) -> dict[str, dict]:
    """스키마별 Pydantic 모델(desc/nodesc)과 rich prompt를 사전 생성.

    Returns:
        {schema_key: {"desc_model": cls, "nodesc_model": cls, "rich_prompt": str, "schema_dict": dict}}
    """
    cache: dict[str, dict] = {}

    for sample in samples:
        schema_dict = adapter.get_schema_dict(sample)
        if adapter.schema_key_fn:
            key = adapter.schema_key_fn(sample)
        else:
            key = sample["id"]

        if key in cache:
            continue

        safe_name = key.replace("/", "_").replace("-", "_").replace(".", "_")
        cache[key] = {
            "desc_model": adapter.schema_fn(
                schema_dict, with_descriptions=True, model_name=f"{safe_name}_Desc"
            ),
            "nodesc_model": adapter.schema_fn(
                schema_dict, with_descriptions=False, model_name=f"{safe_name}_NoDesc"
            ),
            "rich_prompt": adapter.prompt_fn(schema_dict),
            "schema_dict": schema_dict,
        }

    return cache


def _save_framework_results(fw_results: list[dict], output_dir: Path, fw: str, mode: str) -> Path:
    """프레임워크별 결과를 개별 JSON 파일로 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{fw}_{mode}.json"
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fw_results, f, indent=2, ensure_ascii=False, default=str)
    return output_path


async def run_benchmark(
    adapter: DatasetAdapter,
    samples: list[dict],
    fw_modes: list[tuple[str, str]],
    model: str,
    base_url: str,
    api_key: str = "dummy",
    combinations: list[dict] | None = None,
    save_predictions: bool = True,
    output_dir: Path | None = None,
) -> list[dict]:
    """벤치마크 실행.

    Returns:
        all_results: list of result dicts
    """
    combos = combinations or COMBINATIONS
    model_cache = _prepare_models_and_prompts(adapter, samples)

    total_tests = len(samples) * len(fw_modes) * len(combos)
    logger.info(f"{adapter.name.upper()} Benchmark | {len(samples)} samples × {len(combos)} combos × {len(fw_modes)} frameworks = {total_tests} tests | Model: {model}")

    all_results: list[dict] = []
    test_num = 0

    for fw, mode in fw_modes:
        label = f"{fw}/{mode}"
        fw_results: list[dict] = []
        logger.info(f"{'='*50} {label} {'='*50}")

        for combo in combos:
            combo_scores: list[float] = []
            logger.info(f"--- {combo['label']} ({combo['id']}) ---")

            for sample in samples:
                test_num += 1
                sid = sample["id"]
                schema_dict = adapter.get_schema_dict(sample)
                gt = adapter.get_ground_truth(sample)

                # 모델/프롬프트 선택
                if adapter.schema_key_fn:
                    cache_key = adapter.schema_key_fn(sample)
                else:
                    cache_key = sid

                cached = model_cache.get(cache_key)
                if cached:
                    if combo["use_desc"]:
                        model_cls = cached["desc_model"]
                    else:
                        model_cls = cached["nodesc_model"]
                    rich_prompt = cached["rich_prompt"]
                else:
                    # cache miss: 동적 생성
                    model_cls = adapter.schema_fn(
                        schema_dict,
                        with_descriptions=combo["use_desc"],
                        model_name=f"D_{sid}",
                    )
                    rich_prompt = adapter.prompt_fn(schema_dict)

                if combo["use_rich"]:
                    prompt = rich_prompt
                else:
                    prompt = adapter.minimal_prompt

                # 메타 정보 (있으면)
                meta_parts = []
                if "category" in sample:
                    meta_parts.append(f"cat={sample['category']}")
                if "true_depth" in sample:
                    meta_parts.append(f"depth={sample['true_depth']}")
                if "domain" in sample:
                    meta_parts.append(f"dom={sample['domain']}")
                meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

                pct = 0.0
                sc: dict = {}
                predicted: Any = None
                log_prefix = f"[{test_num}/{total_tests}] {label:30s} {combo['id']:8s} {sid}{meta_str}"
                try:
                    r = await _run_single(
                        fw, mode, model_cls, prompt, sample["text"],
                        model, base_url, api_key,
                    )
                    if r["success"] and r["data"]:
                        predicted = r["data"]
                        sc = score_result(predicted, gt, schema_dict)
                        pct = sc["pct"]
                        logger.success(f"{log_prefix} OK {r['latency_ms']:>7.0f}ms  {pct:>5.1f}%")
                    else:
                        logger.warning(f"{log_prefix} FAIL {r['latency_ms']:>5.0f}ms  {(r.get('error') or '')[:40]}")
                except Exception as e:
                    r = {"success": False, "latency_ms": 0, "error": str(e)[:200]}
                    logger.error(f"{log_prefix} ERROR {str(e)[:40]}")

                combo_scores.append(pct)

                result_entry: dict[str, Any] = {
                    "dataset": adapter.name,
                    "combination": combo["id"],
                    "framework": fw,
                    "mode": mode,
                    "sample_id": sid,
                    "success": r["success"],
                    "score_pct": pct,
                    "latency_ms": r["latency_ms"],
                    "error": r.get("error"),
                }
                # 데이터셋별 메타 필드
                for k in ("category", "true_depth", "domain", "schema_name"):
                    if k in sample:
                        result_entry[k] = sample[k]

                if save_predictions:
                    result_entry["ground_truth"] = gt
                    result_entry["predicted"] = predicted
                    result_entry["field_scores"] = sc.get("field_scores", {})

                all_results.append(result_entry)
                fw_results.append(result_entry)

            # 조합별 평균
            ok = [s for s in combo_scores if s > 0]
            avg = sum(ok) / len(ok) if ok else 0
            fail_cnt = len(combo_scores) - len(ok)
            logger.info(f"→ {label:30s} {combo['id']:8s} AVG={avg:>5.1f}%  (fail={fail_cnt}/{len(combo_scores)})")

        # 프레임워크 완료 시 개별 파일 저장
        if output_dir:
            saved = _save_framework_results(fw_results, output_dir, fw, mode)
            logger.success(f"[Saved] {saved} ({len(fw_results)} results)")

    return all_results


def print_summary(all_results: list[dict], fw_modes: list[tuple[str, str]], combos: list[dict] | None = None):
    """터미널에 최종 요약 테이블 출력."""
    combos = combos or COMBINATIONS

    summary_lines = []
    summary_lines.append(f"{'='*90}")
    summary_lines.append(f" FINAL SUMMARY: Average Score by Framework × Combination")
    summary_lines.append(f"{'='*90}")
    header = f"  {'Framework/Mode':<30}"
    for combo in combos:
        header += f" {combo['id'][:12]:>12}"
    header += f" {'Overall':>10}"
    summary_lines.append(header)
    summary_lines.append(f"  {'-'*30}" + f" {'-'*12}" * len(combos) + f" {'-'*10}")

    for fw, mode in fw_modes:
        label = f"{fw}/{mode}"
        row = f"  {label:<30}"
        overall: list[float] = []
        for combo in combos:
            subset = [
                r for r in all_results
                if r["combination"] == combo["id"] and r["framework"] == fw and r["mode"] == mode
            ]
            ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
            fail = len(subset) - len(ok)
            if ok:
                avg = sum(ok) / len(ok)
                overall.extend(ok)
                row += f" {avg:>7.1f}%({fail}F)" if fail else f" {avg:>10.1f}%"
            else:
                row += f" {'ALL FAIL':>12}"
        if overall:
            row += f" {sum(overall)/len(overall):>8.1f}%"
        else:
            row += f" {'N/A':>10}"
        summary_lines.append(row)

    # 조합별 전체 평균
    combo_avg_row = f"\n  {'COMBINATION AVG':<30}"
    for combo in combos:
        subset = [r for r in all_results if r["combination"] == combo["id"]]
        ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
        avg = sum(ok) / len(ok) if ok else 0
        fail = len(subset) - len(ok)
        combo_avg_row += f" {avg:>7.1f}%({fail}F)"
    summary_lines.append(combo_avg_row)

    # breakdown (category 또는 domain)
    for group_key in ("category", "domain"):
        groups = sorted(set(r.get(group_key, "") for r in all_results if r.get(group_key)))
        if not groups:
            continue
        summary_lines.append(f"\n{'='*90}")
        summary_lines.append(f" {group_key.upper()} BREAKDOWN")
        summary_lines.append(f"{'='*90}")
        for g in groups:
            row = f"  {g:<30}"
            for combo in combos:
                subset = [
                    r for r in all_results
                    if r["combination"] == combo["id"] and r.get(group_key) == g
                ]
                ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
                avg = sum(ok) / len(ok) if ok else 0
                fail = len(subset) - len(ok)
                row += f" {avg:>7.1f}%({fail}F)"
            summary_lines.append(row)

    logger.info("\n" + "\n".join(summary_lines))


def save_results(
    all_results: list[dict],
    dataset_name: str,
    output_dir: Path | None = None,
) -> Path:
    """전체 결과를 all.json으로 저장."""
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / f"{dataset_name}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.success(f"All results saved to {output_path}")
    return output_path
