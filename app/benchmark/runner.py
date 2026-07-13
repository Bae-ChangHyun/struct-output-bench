"""통합 벤치마크 러너.

samples × combinations × frameworks 를 동일 로직으로 실행·채점·저장한다.

측정 방법론:
- repeats: 각 셀을 여러 번 실행해 실행 간 변동을 측정(중앙값 latency, 평균 점수, 재현 표준편차).
- interleave: 한 샘플에서 모든 프레임워크를 번갈아 실행해 서버 부하 표류가 특정 프레임워크에
  체계적 유불리로 작용하지 않게 한다.
- warmup: 프레임워크별 첫 호출의 커넥션 콜드스타트를 계측에서 제외.
- per-call timeout: 한 프레임워크가 멈춰도 벤치 전체가 막히지 않게 실패로 기록.
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.frameworks.base import BaseFrameworkAdapter
from app.frameworks.registry import FrameworkRegistry
from app.scoring import score_result
from app.benchmark.config import COMBINATIONS
from app.benchmark.datasets import DatasetAdapter

import app.frameworks  # noqa: F401  (어댑터 자동 등록)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def _mean_ci95_over_cells(scores: list[float]) -> tuple[float, float]:
    """벤치 셀(sample×combo)들의 평균에 대한 95% 신뢰구간(정규근사). (lower, upper).

    주의: 이 CI는 '데이터셋 셀 간 분산' 즉 스위트 수준의 평균 신뢰구간이며,
    동일 입력의 실행 간 재현성이 아니다(그건 repeats 기반 재현 표준편차로 별도 보고).
    """
    n = len(scores)
    if n < 2:
        mean = scores[0] if n == 1 else 0.0
        return (mean, mean)
    mean = statistics.fmean(scores)
    std_err = statistics.stdev(scores) / math.sqrt(n)
    margin = 1.96 * std_err
    # 점수는 [0,100] 유계이므로 정규근사 구간을 클램프한다.
    return (round(max(0.0, mean - margin), 1), round(min(100.0, mean + margin), 1))


async def _run_once(
    adapter: BaseFrameworkAdapter, schema_class: type, system_prompt: str, text: str,
    timeout: float,
) -> dict:
    """단일 추출 1회. timeout 초과 시 실패로 기록해 전체 벤치 중단을 막는다."""
    try:
        result = await asyncio.wait_for(
            adapter.run(text=text, schema_class=schema_class, system_prompt=system_prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {
            "success": False,
            "latency_ms": round(timeout * 1000, 1),
            "data": None,
            "error": f"timeout after {timeout}s",
        }
    return {
        "success": result.success,
        "latency_ms": round(result.latency_ms, 1),
        "data": result.data,
        "error": (result.error or "")[:200] if not result.success else None,
    }


def _prepare_models_and_prompts(
    adapter: DatasetAdapter, samples: list[dict]
) -> dict[str, dict]:
    """스키마별 Pydantic 모델(desc/nodesc)과 rich prompt를 사전 생성해 재사용한다."""
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


def _resolve_model_and_prompt(
    adapter: DatasetAdapter, sample: dict, combo: dict, model_cache: dict[str, dict]
) -> tuple[type, str]:
    """샘플·조합에 맞는 Pydantic 모델과 프롬프트를 선택한다."""
    sid = sample["id"]
    schema_dict = adapter.get_schema_dict(sample)
    cache_key = adapter.schema_key_fn(sample) if adapter.schema_key_fn else sid

    cached = model_cache.get(cache_key)
    if cached:
        model_cls = cached["desc_model"] if combo["use_desc"] else cached["nodesc_model"]
        rich_prompt = cached["rich_prompt"]
    else:
        model_cls = adapter.schema_fn(
            schema_dict, with_descriptions=combo["use_desc"], model_name=f"D_{sid}"
        )
        rich_prompt = adapter.prompt_fn(schema_dict)

    prompt = rich_prompt if combo["use_rich"] else adapter.minimal_prompt
    return model_cls, prompt


def _aggregate_cell(
    runs: list[dict], gt: dict, schema_dict: dict,
    dataset: str, model: str, combo: dict, fw: str, mode: str, sample: dict,
    repeats: int, save_predictions: bool,
) -> dict:
    """한 셀의 반복 실행 결과를 채점·집계해 결과 엔트리를 만든다."""
    scores: list[float] = []
    scores_exact: list[float] = []
    latencies: list[float] = []
    predicted: Any = None
    field_scores: dict = {}
    last_error: str | None = None

    for r in runs:
        if r["success"] and r["data"]:
            sc = score_result(r["data"], gt, schema_dict)
            scores.append(sc["pct"])
            scores_exact.append(sc["pct_exact"])
            latencies.append(r["latency_ms"])
            predicted = r["data"]
            field_scores = sc.get("field_scores", {})
        elif r.get("error"):
            last_error = r["error"]

    n_success = len(scores)
    succeeded = n_success > 0

    entry: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "combination": combo["id"],
        "framework": fw,
        "mode": mode,
        "sample_id": sample["id"],
        "success": succeeded,
        "n_repeats": repeats,
        "n_success": n_success,
        "success_rate": round(n_success / repeats, 3),
        # 점수는 전체 반복 대비 평균(실패 repeat = 0점)이라 신뢰성이 점수에 반영된다.
        # repeats=1이면 기존과 동일. std는 성공분의 품질 일관성.
        "score_pct": round(sum(scores) / repeats, 1),
        "score_pct_exact": round(sum(scores_exact) / repeats, 1),
        "score_pct_std": round(statistics.stdev(scores), 2) if len(scores) >= 2 else 0.0,
        "latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latencies": latencies,
        "error": None if succeeded else (last_error or "all repeats failed"),
    }
    for k in ("category", "true_depth", "domain", "schema_name"):
        if k in sample:
            entry[k] = sample[k]

    if save_predictions:
        entry["ground_truth"] = gt
        entry["predicted"] = predicted
        entry["field_scores"] = field_scores

    return entry


def _save_framework_results(fw_results: list[dict], output_dir: Path, fw: str, mode: str) -> Path:
    """프레임워크별 결과를 개별 JSON 파일로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{fw}--{mode}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fw_results, f, indent=2, ensure_ascii=False, default=str)
    return output_path


async def _warmup(
    adapters: dict[tuple[str, str], BaseFrameworkAdapter],
    adapter: DatasetAdapter, sample: dict, combo: dict,
    model_cache: dict[str, dict], timeout: float,
) -> None:
    """프레임워크별 1회 throwaway 호출로 커넥션 콜드스타트를 계측에서 제외한다."""
    model_cls, prompt = _resolve_model_and_prompt(adapter, sample, combo, model_cache)
    for (fw, mode), fw_adapter in adapters.items():
        r = await _run_once(fw_adapter, model_cls, prompt, sample["text"], timeout)
        status = "ok" if r["success"] else f"fail({(r.get('error') or '')[:30]})"
        logger.debug(f"warmup {fw}/{mode}: {status}")


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
    repeats: int = 1,
    warmup: bool = False,
    per_call_timeout: float | None = None,
) -> list[dict]:
    """벤치마크 실행. 결과 엔트리 리스트를 반환한다.

    한 샘플에서 프레임워크를 번갈아 실행(interleave)해 서버 부하 표류가 특정 프레임워크에
    체계적 유불리로 작용하지 않게 한다.

    Args:
        repeats: 셀당 실행 횟수(>=1). 실행 간 변동 측정에 사용.
        warmup: 프레임워크별 첫 호출 콜드스타트를 계측 제외.
        per_call_timeout: 호출당 타임아웃(초). None이면 어댑터별 기본 timeout 사용.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    combos = combinations or COMBINATIONS
    model_cache = _prepare_models_and_prompts(adapter, samples)

    adapters: dict[tuple[str, str], BaseFrameworkAdapter] = {
        (fw, mode): FrameworkRegistry.get(fw)(model=model, base_url=base_url, api_key=api_key, mode=mode)
        for fw, mode in fw_modes
    }

    total_cells = len(samples) * len(fw_modes) * len(combos)
    total_calls = total_cells * repeats
    logger.info(
        f"{adapter.name.upper()} Benchmark | {len(samples)} samples × {len(combos)} combos × "
        f"{len(fw_modes)} frameworks × {repeats} repeats = {total_calls} calls "
        f"(interleaved, warmup={warmup}) | Model: {model}"
    )

    if warmup and samples:
        await _warmup(adapters, adapter, samples[0], combos[0], model_cache, per_call_timeout or _first_timeout(adapters))

    fw_results_map: dict[tuple[str, str], list[dict]] = {k: [] for k in adapters}
    all_results: list[dict] = []
    cell_num = 0

    for combo in combos:
        logger.info(f"--- {combo['label']} ({combo['id']}) ---")

        for sample in samples:
            sid = sample["id"]
            gt = adapter.get_ground_truth(sample)
            schema_dict = adapter.get_schema_dict(sample)
            model_cls, prompt = _resolve_model_and_prompt(adapter, sample, combo, model_cache)

            cell_runs: dict[tuple[str, str], list[dict]] = {k: [] for k in adapters}
            for _rep in range(repeats):
                for key, fw_adapter in adapters.items():
                    timeout = per_call_timeout or fw_adapter.timeout
                    run = await _run_once(fw_adapter, model_cls, prompt, sample["text"], timeout)
                    cell_runs[key].append(run)

            for (fw, mode) in fw_modes:
                cell_num += 1
                entry = _aggregate_cell(
                    cell_runs[(fw, mode)], gt, schema_dict, adapter.name, model,
                    combo, fw, mode, sample, repeats, save_predictions,
                )
                all_results.append(entry)
                fw_results_map[(fw, mode)].append(entry)
                _log_cell(cell_num, total_cells, fw, mode, combo, sample, entry)

        _log_combo_averages(all_results, fw_modes, combo)

    if output_dir:
        for (fw, mode), fw_results in fw_results_map.items():
            saved = _save_framework_results(fw_results, output_dir, fw, mode)
            logger.success(f"[Saved] {saved} ({len(fw_results)} results)")

    return all_results


def _first_timeout(adapters: dict[tuple[str, str], BaseFrameworkAdapter]) -> float:
    for a in adapters.values():
        return a.timeout
    return 120.0


def _log_cell(cell_num, total_cells, fw, mode, combo, sample, entry):
    label = f"{fw}/{mode}"
    meta_parts = []
    for k, tag in (("category", "cat"), ("true_depth", "depth"), ("domain", "dom")):
        if k in sample:
            meta_parts.append(f"{tag}={sample[k]}")
    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
    prefix = f"[{cell_num}/{total_cells}] {label:30s} {combo['id']:8s} {sample['id']}{meta_str}"
    std = f"±{entry['score_pct_std']:.1f}" if entry["n_repeats"] > 1 else ""

    if entry["success"]:
        rel = "" if entry["n_success"] == entry["n_repeats"] else f" {entry['n_success']}/{entry['n_repeats']}ok"
        logger.success(
            f"{prefix} OK {entry['latency_ms']:>7.0f}ms  "
            f"NED={entry['score_pct']:>5.1f}%{std} exact={entry['score_pct_exact']:>5.1f}%{rel}"
        )
    else:
        logger.warning(f"{prefix} FAIL  {(entry['error'] or '')[:40]}")


def _log_combo_averages(all_results, fw_modes, combo):
    for fw, mode in fw_modes:
        subset = [
            r for r in all_results
            if r["combination"] == combo["id"] and r["framework"] == fw and r["mode"] == mode
        ]
        if not subset:
            continue
        ok = [r["score_pct"] for r in subset if r["success"]]
        all_scores = [r["score_pct"] for r in subset]
        fail_cnt = len(subset) - len(ok)
        avg_ok = statistics.fmean(ok) if ok else 0.0
        avg_all = statistics.fmean(all_scores) if all_scores else 0.0
        logger.info(
            f"→ {fw}/{mode:20s} {combo['id']:8s} "
            f"AVG={avg_ok:>5.1f}% (success only)  "
            f"AVG_ALL={avg_all:>5.1f}% (incl. fail)  "
            f"(fail={fail_cnt}/{len(subset)})"
        )


def print_summary(all_results: list[dict], fw_modes: list[tuple[str, str]], combos: list[dict] | None = None):
    """터미널에 최종 요약 테이블 출력."""
    combos = combos or COMBINATIONS

    models = sorted(set(r.get("model", "") for r in all_results if r.get("model")))
    model_str = ", ".join(models) if models else "unknown"
    repeats = max((r.get("n_repeats", 1) for r in all_results), default=1)

    lines = []
    lines.append(f"{'='*104}")
    lines.append(f" FINAL SUMMARY: Score by Framework × Combination  (NED = 부분점수, Exact = 엄격일치)")
    lines.append(f" Model: {model_str} | repeats={repeats}")
    lines.append(f" 점수 = 전체 반복 대비 평균(실패 repeat=0) | 콤보 셀 = ≥1회 성공 셀만 | Overall·CI·Exact = 전멸(0점) 셀 포함 전체")
    lines.append(f"{'='*104}")
    header = f"  {'Framework/Mode':<30}"
    for combo in combos:
        header += f" {combo['id'][:12]:>12}"
    header += f" {'NED% [95%CI]':>20} {'Exact%':>8}"
    if repeats > 1:
        header += f" {'repro±σ':>8}"
    lines.append(header)
    lines.append(f"  {'-'*30}" + f" {'-'*12}" * len(combos) + f" {'-'*20} {'-'*8}" + (f" {'-'*8}" if repeats > 1 else ""))

    for fw, mode in fw_modes:
        label = f"{fw}/{mode}"
        row = f"  {label:<30}"
        overall: list[float] = []
        overall_exact: list[float] = []
        repro_stds: list[float] = []
        for combo in combos:
            subset = [
                r for r in all_results
                if r["combination"] == combo["id"] and r["framework"] == fw and r["mode"] == mode
            ]
            ok = [r["score_pct"] for r in subset if r["success"]]
            fail = len(subset) - len(ok)
            # Overall/Exact/repro는 실패 셀(=0점)까지 포함해 항상 누적한다. ALL-FAIL 콤보를
            # 제외하면 전멸한 프레임워크가 부분성공한 프레임워크보다 높게 나오는 역전이 생긴다.
            overall.extend(r["score_pct"] for r in subset)
            overall_exact.extend(r["score_pct_exact"] for r in subset)
            repro_stds.extend(r["score_pct_std"] for r in subset if r["success"])
            # 셀 표시는 '성공 시 품질'을 보이도록 success-only 평균.
            if ok:
                avg = statistics.fmean(ok)
                row += f" {avg:>7.1f}%({fail}F)" if fail else f" {avg:>10.1f}%"
            else:
                row += f" {'ALL FAIL':>12}"
        if overall:
            avg_overall = statistics.fmean(overall)
            ci_lo, ci_hi = _mean_ci95_over_cells(overall)
            avg_exact = statistics.fmean(overall_exact) if overall_exact else 0.0
            row += f" {avg_overall:>6.1f}% [{ci_lo:.0f}-{ci_hi:.0f}] {avg_exact:>7.1f}%"
            if repeats > 1:
                mean_std = statistics.fmean(repro_stds) if repro_stds else 0.0
                row += f" {mean_std:>7.2f}"
        else:
            row += f" {'N/A':>20} {'N/A':>8}"
        lines.append(row)

    combo_avg_row = f"\n  {'COMBINATION AVG':<30}"
    for combo in combos:
        subset = [r for r in all_results if r["combination"] == combo["id"]]
        all_scores = [r["score_pct"] for r in subset]
        fail = len(subset) - len([r for r in subset if r["success"]])
        avg = statistics.fmean(all_scores) if all_scores else 0.0
        combo_avg_row += f" {avg:>7.1f}%({fail}F)"
    lines.append(combo_avg_row)

    for group_key in ("category", "domain"):
        groups = sorted(set(r.get(group_key, "") for r in all_results if r.get(group_key)))
        if not groups:
            continue
        lines.append(f"\n{'='*104}")
        lines.append(f" {group_key.upper()} BREAKDOWN (NED%, success only)")
        lines.append(f"{'='*104}")
        for g in groups:
            row = f"  {g:<30}"
            for combo in combos:
                subset = [
                    r for r in all_results
                    if r["combination"] == combo["id"] and r.get(group_key) == g
                ]
                ok = [r["score_pct"] for r in subset if r["success"]]
                avg = statistics.fmean(ok) if ok else 0.0
                fail = len(subset) - len(ok)
                row += f" {avg:>7.1f}%({fail}F)"
            lines.append(row)

    logger.info("\n" + "\n".join(lines))


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
