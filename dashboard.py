"""Structured Output Benchmark 결과 대시보드.

Usage:
    uv run streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

# 조합 ID → 대시보드 표시명
COMBO_DISPLAY = {
    "A_desc": "Prompt(min) + Desc(O)",
    "B_nodesc": "Prompt(min) + Desc(X)",
    "C_rich": "Prompt(rich) + Desc(X)",
}


def _list_run_dirs() -> list[str]:
    """results/ 하위 실행 디렉토리 목록 반환 (최신순)."""
    if not RESULTS_DIR.exists():
        return []
    dirs = []
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if d.is_dir() and any(d.glob("*.json")):
            dirs.append(d.name)
    return dirs


@st.cache_data
def load_results(run_dir_name: str | None = None) -> list[dict]:
    """results/ 하위 디렉토리의 JSON 결과 파일 로드.

    구조: results/{dataset}_{timestamp}/{fw}_{mode}.json + all.json
    """
    all_data = []
    if not RESULTS_DIR.exists():
        return all_data

    if run_dir_name:
        target = RESULTS_DIR / run_dir_name
        search_dirs = [target] if target.exists() else []
    else:
        search_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]

    for d in search_dirs:
        for f in sorted(d.glob("*.json")):
            if f.name in ("all.json", "run_config.json"):
                continue
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, list):
                for item in data:
                    item["_source"] = d.name
                all_data.extend(data)
    return all_data


def main():
    st.set_page_config(page_title="Struct Bench", layout="wide")
    st.title("Structured Output Benchmark Dashboard")

    # 실행 디렉토리 선택 드롭다운
    run_dirs = _list_run_dirs()
    if not run_dirs:
        st.warning(
            "결과 파일이 없습니다. `results/` 디렉토리에 벤치마크 결과 JSON을 넣어주세요.\n\n"
            "```bash\n"
            "uv run python run_benchmark.py --dataset deepjsoneval\n"
            "```"
        )
        return

    sel_run = st.selectbox(
        "Run 선택",
        ["전체 (모든 실행)"] + run_dirs,
        index=0,
    )
    run_dir_name = None if sel_run == "전체 (모든 실행)" else sel_run

    results = load_results(run_dir_name)
    if not results:
        st.warning(
            "결과 파일이 없습니다. `results/` 디렉토리에 벤치마크 결과 JSON을 넣어주세요.\n\n"
            "```bash\n"
            "uv run python run_benchmark.py --dataset deepjsoneval\n"
            "uv run python run_benchmark.py --dataset extractbench\n"
            "```"
        )
        return

    df = pd.DataFrame(results)

    # dataset 컬럼 정규화 (_source fallback)
    if "dataset" not in df.columns:
        df["dataset"] = df["_source"]

    # 조합명을 직관적으로 변환
    if "combination" in df.columns:
        df["combination"] = df["combination"].map(lambda x: COMBO_DISPLAY.get(x, x))

    # ── 사이드바: 필터 ──
    st.sidebar.header("Filters")

    # 데이터셋 선택
    datasets = sorted(df["dataset"].unique())
    sel_datasets = st.sidebar.multiselect("Dataset", datasets, default=datasets)
    df = df[df["dataset"].isin(sel_datasets)]

    # 조합 선택
    combos = sorted(df["combination"].unique())
    sel_combos = st.sidebar.multiselect("Combination", combos, default=combos)
    df = df[df["combination"].isin(sel_combos)]

    # 프레임워크 선택
    df["fw_mode"] = df["framework"] + "/" + df["mode"]
    fw_modes = sorted(df["fw_mode"].unique())
    sel_fws = st.sidebar.multiselect("Framework", fw_modes, default=fw_modes)
    df = df[df["fw_mode"].isin(sel_fws)]

    if df.empty:
        st.info("선택된 필터에 해당하는 결과가 없습니다.")
        return

    # ── 탭 구성 ──
    tab_summary, tab_detail, tab_compare = st.tabs(
        ["Summary", "Detail (GT vs Pred)", "Field Scores"]
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_summary:
        st.subheader("Framework × Combination 평균 점수")

        # 피벗 테이블
        pivot = df.pivot_table(
            index="fw_mode",
            columns="combination",
            values="score_pct",
            aggfunc="mean",
        ).round(1)

        if not pivot.empty:
            pivot["Overall"] = pivot.mean(axis=1).round(1)
            pivot = pivot.sort_values("Overall", ascending=False)

            # 색상 표시
            st.dataframe(
                pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                width='stretch',
                height=400,
            )

        # 성공/실패 통계
        st.subheader("성공/실패 통계")
        fail_stats = df.groupby("fw_mode").agg(
            total=("success", "count"),
            success=("success", "sum"),
            avg_score=("score_pct", "mean"),
            avg_latency=("latency_ms", "mean"),
        ).round(1)
        fail_stats["fail"] = fail_stats["total"] - fail_stats["success"]
        fail_stats["fail_rate"] = ((fail_stats["fail"] / fail_stats["total"]) * 100).round(1)
        st.dataframe(fail_stats.sort_values("avg_score", ascending=False), width='stretch')

        # 카테고리/도메인별 breakdown
        for group_col in ("category", "domain"):
            if group_col in df.columns and df[group_col].notna().any():
                st.subheader(f"{group_col.title()} Breakdown")
                grp_pivot = df[df[group_col].notna()].pivot_table(
                    index=group_col,
                    columns="combination",
                    values="score_pct",
                    aggfunc="mean",
                ).round(1)
                if not grp_pivot.empty:
                    st.dataframe(
                        grp_pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                        width='stretch',
                    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: Detail (GT vs Pred)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_detail:
        st.subheader("개별 샘플 GT vs Predicted 비교")

        # 샘플 선택
        has_gt = "ground_truth" in df.columns and df["ground_truth"].notna().any()
        if not has_gt:
            st.info(
                "예측 결과(ground_truth, predicted)가 저장되지 않은 결과입니다.\n\n"
                "벤치마크 러너를 업데이트하면 GT/Predicted를 저장할 수 있습니다."
            )
        else:
            sample_ids = sorted(df["sample_id"].unique())
            sel_sample = st.selectbox("Sample", sample_ids)
            sample_df = df[df["sample_id"] == sel_sample]

            sel_combo = st.selectbox("Combination", sorted(sample_df["combination"].unique()))
            combo_df = sample_df[sample_df["combination"] == sel_combo]

            for _, row in combo_df.iterrows():
                with st.expander(f"{row['fw_mode']} — {row['score_pct']:.1f}%", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Ground Truth**")
                        gt = row.get("ground_truth")
                        if isinstance(gt, str):
                            gt = json.loads(gt)
                        st.json(gt if gt else {})
                    with col2:
                        st.markdown("**Predicted**")
                        pred = row.get("predicted")
                        if isinstance(pred, str):
                            pred = json.loads(pred)
                        st.json(pred if pred else {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: Field Scores
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_compare:
        st.subheader("필드별 점수 분석")

        has_fields = "field_scores" in df.columns and df["field_scores"].notna().any()
        if not has_fields:
            st.info("field_scores 데이터가 없습니다. 벤치마크 러너 업데이트 후 확인 가능합니다.")
        else:
            sample_ids = sorted(df["sample_id"].unique())
            sel_sample = st.selectbox("Sample ", sample_ids, key="fs_sample")
            sample_df = df[df["sample_id"] == sel_sample]

            sel_combo = st.selectbox("Combination ", sorted(sample_df["combination"].unique()), key="fs_combo")
            combo_df = sample_df[sample_df["combination"] == sel_combo]

            # 프레임워크별 필드 점수 테이블
            rows = []
            for _, row in combo_df.iterrows():
                fs = row.get("field_scores", {})
                if isinstance(fs, str):
                    fs = json.loads(fs)
                if fs:
                    for path, score in fs.items():
                        rows.append({
                            "framework": row["fw_mode"],
                            "field": path,
                            "score": score,
                        })

            if rows:
                fs_df = pd.DataFrame(rows)
                pivot = fs_df.pivot_table(index="field", columns="framework", values="score").round(3)
                pivot["avg"] = pivot.mean(axis=1).round(3)
                pivot = pivot.sort_values("avg")

                st.dataframe(
                    pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                    width='stretch',
                    height=600,
                )
            else:
                st.info("필드 점수 데이터가 없습니다.")


if __name__ == "__main__":
    main()
