"""
Multi-Resume Benchmark: 10개 이력서 × 3가지 조합 × 선택된 프레임워크/모드

속도를 위해 각 조합별 대표 프레임워크/모드만 테스트:
  - instructor/tools (tool calling + desc 자동 포함)
  - instructor/json_schema (json schema mode)
  - openai (native json schema - xgrammar)
  - langchain/json_schema (json schema)
  - langchain/function_calling (tool calling)
  - marvin (tool calling)
  - pydantic_ai (tool calling)
  - mirascope (tool calling)
  - guardrails (litellm)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.frameworks.registry import FrameworkRegistry
from app.schemas import get_schema
from app.prompts.loader import load_prompt

import app.frameworks  # noqa: F401

# ── 설정 ──
BASE_URL = "http://118.38.20.101:8001/v1"
MODEL = "openai/gpt-oss-120b"
API_KEY = "dummy"
RESUME_DIR = Path(__file__).parent / "resumes"

# 테스트할 프레임워크/모드 조합 (핵심만)
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
    {"id": "A_desc", "label": "Schema(desc)+Prompt(min)",
     "schema": "MainInfo", "prompt": "extract_career_minimal"},
    {"id": "B_nodesc", "label": "Schema(nodesc)+Prompt(min)",
     "schema": "MainInfoNoDesc", "prompt": "extract_career_minimal"},
    {"id": "C_rich", "label": "Schema(nodesc)+Prompt(rich)",
     "schema": "MainInfoNoDesc", "prompt": "extract_career_rich"},
]

# ground truth import
from resumes.ground_truths import GROUND_TRUTHS


def score_result(data: dict | None, gt: dict) -> dict:
    """Ground truth 기반 채점 (총 100점)"""
    if not data:
        return {"total": 0, "max": 100, "pct": 0.0}

    scores = {}
    max_score = 0

    # 1. 경력 수 (10점)
    careers = data.get("careers", [])
    max_score += 10
    if len(careers) == gt["career_count"]:
        scores["career_count"] = 10
    elif len(careers) >= gt["career_count"] - 1:
        scores["career_count"] = 6
    elif len(careers) >= 1:
        scores["career_count"] = 3
    else:
        scores["career_count"] = 0

    # 2. 회사명/비공개 매칭 (10점)
    max_score += 10
    company_text = " ".join(c.get("company_name", "") or "" for c in careers)
    company_matches = 0
    for gcomp in gt["companies"]:
        if gcomp.get("is_private"):
            if any(c.get("is_company_private") is True or
                   "비공개" in (c.get("company_name", "") or "")
                   for c in careers):
                company_matches += 1
        else:
            if any(k.lower() in company_text.lower() for k in gcomp["name_keywords"]):
                company_matches += 1
    scores["companies"] = int(company_matches / max(len(gt["companies"]), 1) * 10)

    # 3. 재직중 감지 (5점)
    max_score += 5
    has_current = any(c.get("is_currently_employed") is True for c in careers)
    expected_current = any(c.get("currently_employed") for c in gt["companies"])
    scores["currently_employed"] = 5 if has_current == expected_current else 0

    # 4. 날짜 정확도 (10점)
    max_score += 10
    career_dates_text = " ".join(
        (c.get("start_date", "") or "") + " " + (c.get("end_date", "") or "")
        for c in careers
    )
    date_matches = 0
    date_total = 0
    for gcomp in gt["companies"]:
        if "start" in gcomp:
            date_total += 1
            if gcomp["start"] in career_dates_text:
                date_matches += 1
    scores["dates"] = int(date_matches / max(date_total, 1) * 10)

    # 5. 업무상세 품질 (10점)
    max_score += 10
    work_details = [c.get("work_details", "") or "" for c in careers]
    non_empty = sum(1 for d in work_details if len(d) > 20)
    expected_non_empty = sum(1 for c in gt["companies"] if not c.get("is_private", False))
    scores["work_details"] = int(non_empty / max(expected_non_empty, 1) * 10)

    # 6. 활동/경험 (10점)
    max_score += 10
    activities = data.get("activity_experiences", [])
    act_text = " ".join(
        " ".join(str(v) for v in a.values() if v) for a in activities
    )
    act_score = 0
    if len(activities) == gt["activity_count"]:
        act_score += 4
    elif len(activities) >= max(gt["activity_count"] - 1, 1):
        act_score += 2
    for kw in gt.get("activities_keywords", []):
        if kw.lower() in act_text.lower():
            act_score += 3
    scores["activities"] = min(act_score, 10)

    # 7. 해외경험 (5점)
    max_score += 5
    overseas = data.get("overseas_experiences", [])
    ovs_text = " ".join(
        " ".join(str(v) for v in o.values() if v) for o in overseas
    )
    ovs_score = 0
    if len(overseas) == gt["overseas_count"]:
        ovs_score += 2
    for kw in gt.get("overseas_keywords", []):
        if kw in ovs_text:
            ovs_score += 1
    scores["overseas"] = min(ovs_score, 5)

    # 8. 어학 (10점)
    max_score += 10
    langs = data.get("language_skills", [])
    lang_text = " ".join(
        " ".join(str(v) for v in ls.values() if v) for ls in langs
    )
    lang_score = 0
    if len(langs) >= gt["language_count"]:
        lang_score += 3
    elif len(langs) >= gt["language_count"] - 1:
        lang_score += 1
    for kw in gt.get("lang_keywords", []):
        if kw.lower() in lang_text.lower():
            lang_score += 2
    scores["languages"] = min(lang_score, 10)

    # 9. 자격증 (10점)
    max_score += 10
    certs = data.get("certificates", [])
    cert_text = " ".join(
        " ".join(str(v) for v in c.values() if v) for c in certs
    )
    cert_score = 0
    if len(certs) == gt["certificate_count"]:
        cert_score += 3
    elif len(certs) >= max(gt["certificate_count"] - 1, 1):
        cert_score += 1
    for kw in gt.get("cert_keywords", []):
        if kw.lower() in cert_text.lower():
            cert_score += 2
    scores["certificates"] = min(cert_score, 10)

    # 10. 수상 (5점)
    max_score += 5
    awards = data.get("award_experiences", [])
    award_text = " ".join(
        " ".join(str(v) for v in a.values() if v) for a in awards
    )
    award_score = 0
    if len(awards) == gt["award_count"]:
        award_score += 2
    for kw in gt.get("award_keywords", []):
        if kw.lower() in award_text.lower():
            award_score += 1
    scores["awards"] = min(award_score, 5)

    # 11. 병역 (10점)
    max_score += 10
    mil = data.get("employment_military_info")
    mil_score = 0
    gt_mil = gt.get("military", {})
    if mil and isinstance(mil, dict):
        if gt_mil.get("status") == "해당없음":
            if mil.get("military_status") in ["해당없음", None, ""] or \
               mil.get("military_status") == "면제":
                mil_score += 10
        else:
            if mil.get("military_status") == gt_mil.get("status"):
                mil_score += 3
            if mil.get("military_branch") == gt_mil.get("branch"):
                mil_score += 3
            if mil.get("rank") == gt_mil.get("rank"):
                mil_score += 2
            if mil.get("is_veteran_target") is False:
                mil_score += 1
            if mil.get("is_disabled") is False:
                mil_score += 1
    elif gt_mil.get("status") == "해당없음":
        mil_score += 5  # 없어도 부분 점수
    scores["military"] = mil_score

    # 12. SNS (5점)
    max_score += 5
    sns = data.get("sns")
    sns_score = 0
    if sns and isinstance(sns, dict):
        links = sns.get("sns_links", [])
        if isinstance(links, list):
            link_text = " ".join(links).lower()
            for kw in gt.get("sns_keywords", []):
                if kw in link_text:
                    sns_score += 1
            if len(links) >= gt["sns_count"]:
                sns_score += 1
    scores["sns"] = min(sns_score, 5)

    total = sum(scores.values())
    return {"total": total, "max": max_score, "pct": round(total / max_score * 100, 1), "details": scores}


async def run_single(fw: str, mode: str, schema_name: str, prompt_name: str, text: str) -> dict:
    adapter_cls = FrameworkRegistry.get(fw)
    adapter = adapter_cls(model=MODEL, base_url=BASE_URL, api_key=API_KEY, mode=mode)
    schema_class = get_schema(schema_name)
    prompt = load_prompt(prompt_name)

    result = await adapter.run(text=text, schema_class=schema_class, system_prompt=prompt.system_prompt)

    return {
        "success": result.success,
        "latency_ms": round(result.latency_ms, 1),
        "data": result.data,
        "error": (result.error or "")[:200] if not result.success else None,
    }


async def main():
    # 이력서 로드
    resume_files = sorted(RESUME_DIR.glob("resume_*.md"))
    resumes = {}
    for f in resume_files:
        key = f.stem  # resume_01, resume_02, ...
        resumes[key] = f.read_text()
    print(f"Loaded {len(resumes)} resumes")

    total_tests = len(resumes) * len(FW_MODES) * len(COMBINATIONS)
    print(f"\n{'='*90}")
    print(f" Multi-Resume Benchmark — 10 Resumes × 3 Combos × {len(FW_MODES)} Frameworks")
    print(f" Model: {MODEL} | Total: {total_tests} tests")
    print(f"{'='*90}\n")

    # 결과 저장: results[combo_id][fw/mode][resume_id] = score_pct
    all_results = []
    test_num = 0

    for combo in COMBINATIONS:
        print(f"\n{'='*70}")
        print(f" {combo['label']} (schema={combo['schema']}, prompt={combo['prompt']})")
        print(f"{'='*70}")

        for fw, mode in FW_MODES:
            label = f"{fw}/{mode}"
            fw_scores = []

            for resume_id, text in resumes.items():
                test_num += 1
                gt = GROUND_TRUTHS.get(resume_id, {})
                print(f"[{test_num}/{total_tests}] {label:30s} {resume_id}", end=" ", flush=True)

                try:
                    r = await run_single(fw, mode, combo["schema"], combo["prompt"], text)
                    if r["success"] and r["data"]:
                        score = score_result(r["data"], gt)
                        pct = score["pct"]
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
                    "resume_id": resume_id,
                    "success": r["success"],
                    "score_pct": pct,
                    "latency_ms": r["latency_ms"],
                })

            # 프레임워크별 평균
            ok_scores = [s for s in fw_scores if s > 0]
            avg = sum(ok_scores) / len(ok_scores) if ok_scores else 0
            fail_cnt = len([s for s in fw_scores if s == 0])
            print(f"  → {label:30s} AVG={avg:>5.1f}%  (fail={fail_cnt}/{len(fw_scores)})\n")

    # ── 최종 요약: 프레임워크 × 조합 매트릭스 ──
    print(f"\n{'='*90}")
    print(f" FINAL SUMMARY: Average Score by Framework × Combination")
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
            subset = [r for r in all_results
                      if r["combination"] == combo["id"]
                      and r["framework"] == fw and r["mode"] == mode]
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

    # JSON 저장
    output_path = Path(__file__).parent / "multi_resume_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
