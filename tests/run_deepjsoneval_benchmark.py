"""
DeepJSONEval Benchmark: 실제 HuggingFace 데이터셋 x 3가지 조합 x 10개 프레임워크/모드

HuggingFace GTSAIInfraLabSOTAS/DeepJSONEval (2,100 rows) 데이터셋을 사용.
각 인스턴스마다 다른 JSON Schema를 가지므로:
- 동적으로 Pydantic 모델 생성 (desc / no_desc)
- 동적으로 rich 프롬프트 생성
- 범용 채점 함수로 ground truth 비교

3가지 조합:
  A_desc:   Pydantic 모델(description 포함) + minimal 프롬프트
  B_nodesc: Pydantic 모델(description 제거) + minimal 프롬프트
  C_rich:   Pydantic 모델(description 제거) + rich 프롬프트(description을 프롬프트에 주입)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.frameworks.registry import FrameworkRegistry
from app.prompts.loader import load_prompt
from app.scoring import score_result
from app.datasets.deepjsoneval import (
    load_samples,
    json_schema_to_pydantic,
    generate_rich_prompt,
)

import app.frameworks  # noqa: F401

# ── 설정 ──
BASE_URL = "http://118.38.20.101:8001/v1"
MODEL = "openai/gpt-oss-120b"
API_KEY = "dummy"

# 테스트할 프레임워크/모드 조합
FW_MODES = [
    ("instructor", "tools"),
    ("instructor", "json_schema"),
    ("openai", "default"),
    ("langchain", "json_schema"),
    ("langchain", "function_calling"),
    ("marvin", "default"),
    ("pydantic_ai", "default"),
    ("mirascope", "default"),
    ("guardrails", "default"),
    ("llamaindex", "default"),
]

# 3가지 조합
COMBINATIONS = [
    {"id": "A_desc", "label": "Schema(desc)+Prompt(min)", "use_desc": True, "use_rich": False},
    {"id": "B_nodesc", "label": "Schema(nodesc)+Prompt(min)", "use_desc": False, "use_rich": False},
    {"id": "C_rich", "label": "Schema(nodesc)+Prompt(rich)", "use_desc": False, "use_rich": True},
]

# 샘플 설정
MAX_SAMPLES = 50
PER_CATEGORY = 5  # 카테고리당 최대 5개 -> 10 카테고리 x 5 = 50


async def run_single(
    fw: str,
    mode: str,
    schema_class,
    system_prompt: str,
    text: str,
) -> dict:
    adapter_cls = FrameworkRegistry.get(fw)
    adapter = adapter_cls(model=MODEL, base_url=BASE_URL, api_key=API_KEY, mode=mode)

    result = await adapter.run(
        text=text, schema_class=schema_class, system_prompt=system_prompt,
    )

    return {
        "success": result.success,
        "latency_ms": round(result.latency_ms, 1),
        "data": result.data,
        "error": (result.error or "")[:200] if not result.success else None,
    }


async def main():
    # 데이터셋 로드 (실제 HuggingFace 데이터)
    print("Loading DeepJSONEval samples from HuggingFace...")
    samples = load_samples(max_samples=MAX_SAMPLES, per_category=PER_CATEGORY)
    print(f"Loaded {len(samples)} samples")

    # 카테고리 분포
    cat_dist: dict[str, int] = {}
    for s in samples:
        cat_dist[s["category"]] = cat_dist.get(s["category"], 0) + 1
    print(f"Categories: {cat_dist}")

    # depth 분포
    depth_dist: dict[int, int] = {}
    for s in samples:
        depth_dist[s["true_depth"]] = depth_dist.get(s["true_depth"], 0) + 1
    print(f"Depths: {depth_dist}")

    # minimal 프롬프트 로드
    minimal_prompt = load_prompt("extract_deepjsoneval_minimal").system_prompt

    total_tests = len(samples) * len(FW_MODES) * len(COMBINATIONS)
    print(f"\n{'='*90}")
    print(f" DeepJSONEval Benchmark -- {len(samples)} Samples x 3 Combos x {len(FW_MODES)} Frameworks")
    print(f" Model: {MODEL} | Total: {total_tests} tests")
    print(f"{'='*90}\n")

    all_results = []
    test_num = 0

    for combo in COMBINATIONS:
        print(f"\n{'='*70}")
        print(f" {combo['label']} (desc={combo['use_desc']}, rich={combo['use_rich']})")
        print(f"{'='*70}")

        for fw, mode in FW_MODES:
            label = f"{fw}/{mode}"
            fw_scores = []

            for sample in samples:
                test_num += 1
                sid = sample["id"]
                schema_dict = sample["schema"]
                gt = sample["ground_truth"]

                # 동적 Pydantic 모델 생성
                model_cls = json_schema_to_pydantic(
                    schema_dict,
                    with_descriptions=combo["use_desc"],
                    model_name=f"D_{sid}",
                )

                # 프롬프트 결정
                if combo["use_rich"]:
                    prompt = generate_rich_prompt(schema_dict)
                else:
                    prompt = minimal_prompt

                print(
                    f"[{test_num}/{total_tests}] {label:30s} {sid} "
                    f"(cat={sample['category']}, depth={sample['true_depth']})",
                    end=" ",
                    flush=True,
                )

                try:
                    r = await run_single(fw, mode, model_cls, prompt, sample["text"])
                    if r["success"] and r["data"]:
                        sc = score_result(r["data"], gt, schema_dict)
                        pct = sc["pct"]
                        print(f"OK {r['latency_ms']:>7.0f}ms  {pct:>5.1f}%")
                    else:
                        pct = 0.0
                        print(f"FAIL {r['latency_ms']:>5.0f}ms  {(r['error'] or '')[:40]}")
                except Exception as e:
                    pct = 0.0
                    r = {"success": False, "latency_ms": 0, "error": str(e)[:200]}
                    print(f"ERROR {str(e)[:40]}")

                fw_scores.append(pct)
                all_results.append({
                    "combination": combo["id"],
                    "framework": fw,
                    "mode": mode,
                    "sample_id": sid,
                    "category": sample["category"],
                    "true_depth": sample["true_depth"],
                    "success": r["success"],
                    "score_pct": pct,
                    "latency_ms": r["latency_ms"],
                })

            # 프레임워크별 평균
            ok_scores = [s for s in fw_scores if s > 0]
            avg = sum(ok_scores) / len(ok_scores) if ok_scores else 0
            fail_cnt = len([s for s in fw_scores if s == 0])
            print(f"  -> {label:30s} AVG={avg:>5.1f}%  (fail={fail_cnt}/{len(fw_scores)})\n")

    # ── 최종 요약: 프레임워크 x 조합 매트릭스 ──
    print(f"\n{'='*90}")
    print(f" FINAL SUMMARY: Average Score by Framework x Combination")
    print(f"{'='*90}")
    header = f"  {'Framework/Mode':<30}"
    for combo in COMBINATIONS:
        header += f" {combo['id'][:12]:>12}"
    header += f" {'Overall':>10}"
    print(header)
    print(f"  {'-'*30}" + f" {'-'*12}" * len(COMBINATIONS) + f" {'-'*10}")

    for fw, mode in FW_MODES:
        label = f"{fw}/{mode}"
        row = f"  {label:<30}"
        overall_scores = []
        for combo in COMBINATIONS:
            subset = [
                r for r in all_results
                if r["combination"] == combo["id"]
                and r["framework"] == fw and r["mode"] == mode
            ]
            ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
            fail = len(subset) - len(ok)
            if ok:
                avg = sum(ok) / len(ok)
                overall_scores.extend(ok)
                if fail > 0:
                    row += f" {avg:>7.1f}%({fail}F)"
                else:
                    row += f" {avg:>10.1f}%"
            else:
                row += f" {'ALL FAIL':>12}"
        if overall_scores:
            row += f" {sum(overall_scores)/len(overall_scores):>8.1f}%"
        else:
            row += f" {'N/A':>10}"
        print(row)

    # ── 조합별 전체 평균 ──
    print(f"\n  {'COMBINATION AVG':<30}", end="")
    for combo in COMBINATIONS:
        subset = [r for r in all_results if r["combination"] == combo["id"]]
        ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
        avg = sum(ok) / len(ok) if ok else 0
        fail = len(subset) - len(ok)
        print(f" {avg:>7.1f}%({fail}F)", end="")
    print()

    # ── 카테고리별 breakdown ──
    print(f"\n{'='*90}")
    print(f" CATEGORY BREAKDOWN (all frameworks avg, by combination)")
    print(f"{'='*90}")

    all_cats = sorted(set(r["category"] for r in all_results))
    cat_header = f"  {'Category':<20}"
    for combo in COMBINATIONS:
        cat_header += f" {combo['id'][:12]:>12}"
    print(cat_header)
    print(f"  {'-'*20}" + f" {'-'*12}" * len(COMBINATIONS))

    for cat in all_cats:
        row = f"  {cat:<20}"
        for combo in COMBINATIONS:
            subset = [
                r for r in all_results
                if r["combination"] == combo["id"] and r["category"] == cat
            ]
            ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
            avg = sum(ok) / len(ok) if ok else 0
            fail = len(subset) - len(ok)
            row += f" {avg:>7.1f}%({fail}F)"
        print(row)

    # ── Depth별 breakdown ──
    print(f"\n{'='*90}")
    print(f" DEPTH BREAKDOWN (all frameworks avg, by combination)")
    print(f"{'='*90}")

    all_depths = sorted(set(r["true_depth"] for r in all_results))
    depth_header = f"  {'Depth':<20}"
    for combo in COMBINATIONS:
        depth_header += f" {combo['id'][:12]:>12}"
    print(depth_header)
    print(f"  {'-'*20}" + f" {'-'*12}" * len(COMBINATIONS))

    for depth in all_depths:
        row = f"  depth={depth:<14}"
        for combo in COMBINATIONS:
            subset = [
                r for r in all_results
                if r["combination"] == combo["id"] and r["true_depth"] == depth
            ]
            ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
            avg = sum(ok) / len(ok) if ok else 0
            fail = len(subset) - len(ok)
            row += f" {avg:>7.1f}%({fail}F)"
        print(row)

    # ── JSON 저장 ──
    output_path = Path(__file__).parent / "deepjsoneval_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
