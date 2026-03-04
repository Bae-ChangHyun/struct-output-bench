"""
ExtractBench Benchmark: 35개 PDF × 3가지 조합 × 선택된 프레임워크/모드

실제 ExtractBench 데이터셋(GitHub)을 사용한 PDF→구조화 추출 벤치마크.
- 5개 스키마 (research, 10kq, credit_agreement, resume, swimming)
- 4개 도메인 (academic, finance, hiring, sport)
- 3가지 조합: A_desc, B_nodesc, C_rich
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.frameworks.registry import FrameworkRegistry
from app.prompts.loader import load_prompt
from app.datasets.extractbench.loader import load_samples
from app.datasets.extractbench.schema_converter import json_schema_to_pydantic
from app.datasets.extractbench.prompt_generator import generate_rich_prompt
from app.datasets.extractbench.scorer import score_result

import app.frameworks  # noqa: F401

# -- 설정 --
BASE_URL = "http://118.38.20.101:8001/v1"
MODEL = "openai/gpt-oss-120b"
API_KEY = "dummy"

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

MINIMAL_PROMPT_NAME = "extract_extractbench_minimal"


async def run_single(
    fw: str, mode: str, schema_class, system_prompt: str, text: str
) -> dict:
    adapter_cls = FrameworkRegistry.get(fw)
    adapter = adapter_cls(model=MODEL, base_url=BASE_URL, api_key=API_KEY, mode=mode)
    result = await adapter.run(
        text=text, schema_class=schema_class, system_prompt=system_prompt
    )
    return {
        "success": result.success,
        "latency_ms": round(result.latency_ms, 1),
        "data": result.data,
        "error": (result.error or "")[:200] if not result.success else None,
    }


async def main():
    # 1. 데이터 로드
    print("Loading ExtractBench samples...")
    samples = load_samples()
    print(f"Loaded {len(samples)} samples")

    # 2. 스키마별 그룹핑
    schema_groups: dict[str, list[dict]] = {}
    schema_dicts: dict[str, dict] = {}
    for s in samples:
        key = f"{s['domain']}/{s['schema_name']}"
        schema_groups.setdefault(key, []).append(s)
        schema_dicts[key] = s["schema_dict"]

    print(f"Schema groups: {list(schema_groups.keys())}")
    print(f"Samples per group: {', '.join(f'{k}={len(v)}' for k, v in schema_groups.items())}")

    # 3. 각 스키마별로 Pydantic 모델 + rich 프롬프트 사전 생성
    pydantic_models_desc: dict[str, type] = {}
    pydantic_models_nodesc: dict[str, type] = {}
    rich_prompts: dict[str, str] = {}

    for key, schema_dict in schema_dicts.items():
        safe_name = key.replace("/", "_").title().replace("_", "")
        pydantic_models_desc[key] = json_schema_to_pydantic(
            schema_dict, with_descriptions=True, model_name=f"{safe_name}Desc"
        )
        pydantic_models_nodesc[key] = json_schema_to_pydantic(
            schema_dict, with_descriptions=False, model_name=f"{safe_name}NoDesc"
        )
        rich_prompts[key] = generate_rich_prompt(schema_dict)

    # 4. 프롬프트 로드
    minimal_prompt = load_prompt(MINIMAL_PROMPT_NAME)

    # 5. 3가지 조합 정의
    combinations = [
        {
            "id": "A_desc",
            "label": "Schema(desc)+Prompt(min)",
            "get_schema": lambda key: pydantic_models_desc[key],
            "get_prompt": lambda _key: minimal_prompt.system_prompt,
        },
        {
            "id": "B_nodesc",
            "label": "Schema(nodesc)+Prompt(min)",
            "get_schema": lambda key: pydantic_models_nodesc[key],
            "get_prompt": lambda _key: minimal_prompt.system_prompt,
        },
        {
            "id": "C_rich",
            "label": "Schema(nodesc)+Prompt(rich)",
            "get_schema": lambda key: pydantic_models_nodesc[key],
            "get_prompt": lambda key: rich_prompts[key],
        },
    ]

    total_tests = len(samples) * len(FW_MODES) * len(combinations)
    print(f"\n{'='*90}")
    print(f" ExtractBench Benchmark")
    print(f" {len(samples)} PDFs x 3 Combos x {len(FW_MODES)} Frameworks = {total_tests} tests")
    print(f" Model: {MODEL}")
    print(f"{'='*90}\n")

    all_results = []
    test_num = 0

    for combo in combinations:
        print(f"\n{'='*70}")
        print(f" {combo['label']} ({combo['id']})")
        print(f"{'='*70}")

        for fw, mode in FW_MODES:
            label = f"{fw}/{mode}"
            fw_scores = []

            for sample in samples:
                test_num += 1
                schema_key = f"{sample['domain']}/{sample['schema_name']}"
                schema_class = combo["get_schema"](schema_key)
                prompt = combo["get_prompt"](schema_key)

                print(
                    f"[{test_num}/{total_tests}] {label:30s} {sample['id']}",
                    end=" ",
                    flush=True,
                )

                try:
                    r = await run_single(fw, mode, schema_class, prompt, sample["text"])
                    if r["success"] and r["data"]:
                        sc = score_result(
                            r["data"], sample["ground_truth"], sample["schema_dict"]
                        )
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
                    "sample_id": sample["id"],
                    "domain": sample["domain"],
                    "schema_name": sample["schema_name"],
                    "success": r["success"],
                    "score_pct": pct,
                    "latency_ms": r["latency_ms"],
                })

            ok_scores = [s for s in fw_scores if s > 0]
            avg = sum(ok_scores) / len(ok_scores) if ok_scores else 0
            fail_cnt = len([s for s in fw_scores if s == 0])
            print(f"  -> {label:30s} AVG={avg:>5.1f}%  (fail={fail_cnt}/{len(fw_scores)})\n")

    # -- 최종 요약 --
    print(f"\n{'='*90}")
    print(f" FINAL SUMMARY: Average Score by Framework x Combination")
    print(f"{'='*90}")
    header = f"  {'Framework/Mode':<30}"
    for combo in combinations:
        header += f" {combo['id'][:12]:>12}"
    header += f" {'Overall':>10}"
    print(header)
    print(f"  {'-'*30}" + f" {'-'*12}" * len(combinations) + f" {'-'*10}")

    for fw, mode in FW_MODES:
        label = f"{fw}/{mode}"
        row = f"  {label:<30}"
        overall_scores = []
        for combo in combinations:
            subset = [
                r
                for r in all_results
                if r["combination"] == combo["id"]
                and r["framework"] == fw
                and r["mode"] == mode
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

    # -- 도메인별 breakdown --
    print(f"\n{'='*90}")
    print(f" DOMAIN BREAKDOWN (all frameworks avg, by combination)")
    print(f"{'='*90}")
    domains = sorted(set(r["domain"] for r in all_results))
    for domain in domains:
        row = f"  {domain:<30}"
        for combo in combinations:
            subset = [
                r
                for r in all_results
                if r["combination"] == combo["id"] and r["domain"] == domain
            ]
            ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
            avg = sum(ok) / len(ok) if ok else 0
            fail = len(subset) - len(ok)
            row += f" {avg:>7.1f}%({fail}F)"
        print(row)

    # -- 조합별 전체 평균 --
    print(f"\n  {'COMBINATION AVG':<30}", end="")
    for combo in combinations:
        subset = [r for r in all_results if r["combination"] == combo["id"]]
        ok = [r["score_pct"] for r in subset if r["score_pct"] > 0]
        avg = sum(ok) / len(ok) if ok else 0
        fail = len(subset) - len(ok)
        print(f" {avg:>7.1f}%({fail}F)", end="")
    print()

    # JSON 저장
    output_path = Path(__file__).parent / "extractbench_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
