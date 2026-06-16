"""
Fill selected gaps in the first-pass seven-cohort harmonized core table.

This script keeps the original outputs unchanged and writes a gap-filled copy
under 03_analysis_pipeline/outputs_gap_filled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PIPELINE_DIR.parent
DEFAULT_INPUT = PIPELINE_DIR / "outputs" / "harmonized_core.parquet"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs_gap_filled"

DATA_ROOT = Path(r"F:\目前养老官方数据库\数据库新20260615\harmonized数据")
SHARE_FILE = DATA_ROOT / "SHARE_Harmonized数据" / "Gateway Harmonized SHARE G" / "GH_SHARE_g.dta"
MHAS_FILE = DATA_ROOT / "MHAS_Harmonized数据" / "Harmonized MHAS (Version C.2)" / "H_MHAS_c2.dta"
KLOSA_FILE = DATA_ROOT / "KLoSA_Harmonized数据" / "H_KLoSA_e3.dta"
LASI_FILE = DATA_ROOT / "LASI_Harmonized数据" / "Harmonized LASI A.3_Stata" / "H_LASI_a3.dta"


sys.path.insert(0, str(SCRIPT_DIR))
from harmonize_and_analyze_aging import add_derived_variables, build_descriptives  # noqa: E402


def id_as_string(series: pd.Series) -> pd.Series:
    """Normalize Stata numeric IDs to the string form used in harmonized_core."""
    out = series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    whole = numeric.notna() & np.isfinite(numeric) & np.isclose(numeric, np.round(numeric))
    if whole.any():
        out.loc[whole] = numeric.loc[whole].round().astype("Int64").astype("string")
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.mask(out.isna() | out.isin(["", "nan", "NaN", "<NA>"]))
    return out


def numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def binary_01(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = numeric(series)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[s.eq(1)] = 1.0
    out.loc[s.eq(0)] = 0.0
    return out


def binary_any(items: list[pd.Series]) -> pd.Series:
    valid = [x for x in items if x is not None and len(x) > 0]
    if not valid:
        return pd.Series(dtype="float64")
    mat = pd.concat(valid, axis=1)
    out = pd.Series(np.nan, index=mat.index, dtype="float64")
    out.loc[mat.eq(1).any(axis=1)] = 1.0
    out.loc[mat.notna().any(axis=1) & ~mat.eq(1).any(axis=1)] = 0.0
    return out


def inverse_binary(series: pd.Series | None) -> pd.Series:
    yes_no = binary_01(series)
    out = pd.Series(np.nan, index=yes_no.index, dtype="float64")
    out.loc[yes_no.eq(0)] = 1.0
    out.loc[yes_no.eq(1)] = 0.0
    return out


def read_stata_existing(path: Path, columns: list[str]) -> pd.DataFrame:
    reader = pd.read_stata(path, iterator=True)
    first_row = reader.read(nrows=1)
    existing = [col for col in columns if col in first_row.columns]
    if not existing:
        raise ValueError(f"No requested columns found in {path}")
    return pd.read_stata(path, columns=existing, convert_categoricals=False)


def concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    cleaned = [frame.dropna(axis=1, how="all") for frame in frames]
    return pd.concat(cleaned, ignore_index=True).reindex(columns=columns)


def wave_label(wave: int, cohort: str) -> str:
    return "wave1" if cohort == "LASI" and wave == 1 else f"第{wave}轮"


def build_share_depression() -> pd.DataFrame:
    waves = [1, 2, 4, 5, 6, 7, 8, 9]
    cols = ["mergeid"] + [f"r{w}eurod" for w in waves]
    raw = read_stata_existing(SHARE_FILE, cols)
    frames = []
    for w in waves:
        col = f"r{w}eurod"
        if col not in raw.columns:
            continue
        score = numeric(raw[col])
        score = score.where(score.between(0, 12))
        tmp = pd.DataFrame(
            {
                "cohort": "SHARE",
                "participant_id": id_as_string(raw["mergeid"]),
                "wave": wave_label(w, "SHARE"),
                "depression_score_fill": score,
                "depressive_symptoms_fill": np.where(score.notna(), (score >= 4).astype(float), np.nan),
            }
        )
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["participant_id"]).drop_duplicates(["cohort", "participant_id", "wave"])


def build_mhas_social() -> pd.DataFrame:
    waves = [3, 4, 5]
    cols = ["rahhidnp"] + [f"r{w}socwk" for w in waves]
    raw = read_stata_existing(MHAS_FILE, cols)
    frames = []
    for w in waves:
        col = f"r{w}socwk"
        if col not in raw.columns:
            continue
        tmp = pd.DataFrame(
            {
                "cohort": "MHAS",
                "participant_id": id_as_string(raw["rahhidnp"]),
                "wave": wave_label(w, "MHAS"),
                "social_participation_fill": binary_01(raw[col]),
            }
        )
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["participant_id"]).drop_duplicates(["cohort", "participant_id", "wave"])


def coalesce_columns(raw: pd.DataFrame, names: list[str]) -> pd.Series:
    pieces = [numeric(raw[name]) for name in names if name in raw.columns]
    if not pieces:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.concat(pieces, axis=1).bfill(axis=1).iloc[:, 0].astype("float64")


def empty_numeric(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def exhaustion_from_binary(raw: pd.DataFrame, primary: str, alternates: list[str]) -> pd.Series:
    pieces = []
    if primary in raw.columns:
        pieces.append(binary_01(raw[primary]))
    for name in alternates:
        if name in raw.columns:
            pieces.append(binary_01(raw[name]))
    return binary_any(pieces)


def exhaustion_from_frequency(raw: pd.DataFrame, names: list[str], threshold: float = 3.0) -> pd.Series:
    pieces = []
    for name in names:
        if name not in raw.columns:
            continue
        values = numeric(raw[name])
        out = pd.Series(np.nan, index=raw.index, dtype="float64")
        out.loc[values.ge(threshold)] = 1.0
        out.loc[values.lt(threshold)] = 0.0
        pieces.append(out)
    return binary_any(pieces)


def low_activity_lasi(raw: pd.DataFrame) -> pd.Series:
    vigorous = numeric(raw["r1vgactx"]) if "r1vgactx" in raw.columns else pd.Series(np.nan, index=raw.index)
    moderate = numeric(raw["r1mdactx"]) if "r1mdactx" in raw.columns else pd.Series(np.nan, index=raw.index)
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    any_activity = vigorous.between(1, 4) | moderate.between(1, 4)
    no_activity = vigorous.eq(5) & moderate.eq(5)
    out.loc[any_activity] = 0.0
    out.loc[no_activity] = 1.0
    return out


def build_mhas_frailty_candidates() -> pd.DataFrame:
    waves = [1, 2, 3, 4, 5]
    suffixes = ["bmi", "mbmi", "fatigue", "effort", "ftired", "walkra", "walk1a", "vigact", "gripsum", "wspeed"]
    cols = ["rahhidnp"] + [f"r{w}{s}" for w in waves for s in suffixes]
    raw = read_stata_existing(MHAS_FILE, cols)
    frames = []
    for w in waves:
        tmp = pd.DataFrame(
            {
                "cohort": "MHAS",
                "participant_id": id_as_string(raw["rahhidnp"]),
                "wave": wave_label(w, "MHAS"),
                "bmi": coalesce_columns(raw, [f"r{w}mbmi", f"r{w}bmi"]),
                "exhaustion": exhaustion_from_binary(raw, f"r{w}fatigue", [f"r{w}effort", f"r{w}ftired"]),
                "mobility_limitation": binary_any(
                    [
                        binary_01(raw[f"r{w}walkra"]) if f"r{w}walkra" in raw.columns else empty_numeric(raw.index),
                        binary_01(raw[f"r{w}walk1a"]) if f"r{w}walk1a" in raw.columns else empty_numeric(raw.index),
                    ]
                ),
                "low_activity": inverse_binary(raw[f"r{w}vigact"]) if f"r{w}vigact" in raw.columns else empty_numeric(raw.index),
                "grip_strength": numeric(raw[f"r{w}gripsum"]) if f"r{w}gripsum" in raw.columns else empty_numeric(raw.index),
                "walk_time_seconds": numeric(raw[f"r{w}wspeed"]) if f"r{w}wspeed" in raw.columns else empty_numeric(raw.index),
            }
        )
        frames.append(tmp)
    out = concat_frames(
        frames,
        [
            "cohort",
            "participant_id",
            "wave",
            "bmi",
            "exhaustion",
            "mobility_limitation",
            "low_activity",
            "grip_strength",
            "walk_time_seconds",
        ],
    )
    return out.dropna(subset=["participant_id"]).drop_duplicates(["cohort", "participant_id", "wave"])


def build_klosa_frailty_candidates() -> pd.DataFrame:
    waves = range(1, 9)
    suffixes = ["bmi", "effortl", "goingl", "vigact", "gripsum", "bedb_k"]
    cols = ["pid"] + [f"r{w}{s}" for w in waves for s in suffixes]
    raw = read_stata_existing(KLOSA_FILE, cols)
    frames = []
    for w in waves:
        tmp = pd.DataFrame(
            {
                "cohort": "KLoSA",
                "participant_id": id_as_string(raw["pid"]),
                "wave": wave_label(w, "KLoSA"),
                "bmi": coalesce_columns(raw, [f"r{w}bmi"]),
                "exhaustion": exhaustion_from_frequency(raw, [f"r{w}effortl", f"r{w}goingl"], threshold=3.0),
                "mobility_limitation": binary_01(raw[f"r{w}bedb_k"]) if f"r{w}bedb_k" in raw.columns else empty_numeric(raw.index),
                "low_activity": inverse_binary(raw[f"r{w}vigact"]) if f"r{w}vigact" in raw.columns else empty_numeric(raw.index),
                "grip_strength": numeric(raw[f"r{w}gripsum"]) if f"r{w}gripsum" in raw.columns else empty_numeric(raw.index),
                "walk_time_seconds": empty_numeric(raw.index),
            }
        )
        frames.append(tmp)
    out = concat_frames(
        frames,
        [
            "cohort",
            "participant_id",
            "wave",
            "bmi",
            "exhaustion",
            "mobility_limitation",
            "low_activity",
            "grip_strength",
            "walk_time_seconds",
        ],
    )
    return out.dropna(subset=["participant_id"]).drop_duplicates(["cohort", "participant_id", "wave"])


def build_lasi_frailty_candidates() -> pd.DataFrame:
    cols = [
        "prim_key",
        "r1mbmi",
        "r1fatigue",
        "r1walkra",
        "r1walk100a",
        "r1gripsum",
        "r1wspeed",
        "r1vgactx",
        "r1mdactx",
    ]
    raw = read_stata_existing(LASI_FILE, cols)
    out = pd.DataFrame(
        {
            "cohort": "LASI",
            "participant_id": id_as_string(raw["prim_key"]),
            "wave": "wave1",
            "bmi": coalesce_columns(raw, ["r1mbmi"]),
            "exhaustion": binary_01(raw["r1fatigue"]) if "r1fatigue" in raw.columns else empty_numeric(raw.index),
            "mobility_limitation": binary_any(
                [
                    binary_01(raw["r1walkra"]) if "r1walkra" in raw.columns else empty_numeric(raw.index),
                    binary_01(raw["r1walk100a"]) if "r1walk100a" in raw.columns else empty_numeric(raw.index),
                ]
            ),
            "low_activity": low_activity_lasi(raw),
            "grip_strength": numeric(raw["r1gripsum"]) if "r1gripsum" in raw.columns else empty_numeric(raw.index),
            "walk_time_seconds": numeric(raw["r1wspeed"]) if "r1wspeed" in raw.columns else empty_numeric(raw.index),
        }
    )
    return out.dropna(subset=["participant_id"]).drop_duplicates(["cohort", "participant_id", "wave"])


def grouped_quantile(values: pd.Series, q: float) -> float:
    valid = values.dropna()
    if valid.empty:
        return np.nan
    return float(valid.quantile(q))


def add_frailty_scores(core: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    keys = core.reset_index()[["index", "cohort", "participant_id", "wave", "female"]]
    merged = keys.merge(candidates, how="left", on=["cohort", "participant_id", "wave"], validate="many_to_one")
    merged["sex_group"] = merged["female"].map({0.0: "male", 1.0: "female"}).fillna("unknown")

    merged["low_bmi"] = np.where(merged["bmi"].notna(), (merged["bmi"] < 18.5).astype(float), np.nan)

    grip_threshold = merged.groupby(["cohort", "wave", "sex_group"])["grip_strength"].transform(
        lambda s: grouped_quantile(s, 0.2)
    )
    merged["weakness"] = np.where(
        merged["grip_strength"].notna() & grip_threshold.notna(),
        (merged["grip_strength"] <= grip_threshold).astype(float),
        np.nan,
    )

    walk_threshold = merged.groupby(["cohort", "wave", "sex_group"])["walk_time_seconds"].transform(
        lambda s: grouped_quantile(s, 0.8)
    )
    slow_by_time = pd.Series(np.nan, index=merged.index, dtype="float64")
    slow_by_time.loc[merged["walk_time_seconds"].notna() & walk_threshold.notna()] = (
        merged.loc[merged["walk_time_seconds"].notna() & walk_threshold.notna(), "walk_time_seconds"]
        >= walk_threshold.loc[merged["walk_time_seconds"].notna() & walk_threshold.notna()]
    ).astype(float)
    merged["slowness"] = binary_any([merged["mobility_limitation"], slow_by_time])

    component_cols = ["low_bmi", "exhaustion", "weakness", "slowness", "low_activity"]
    components = merged[component_cols]
    merged["frailty_components_available"] = components.notna().sum(axis=1)
    merged["frailty_components_positive"] = components.sum(axis=1, min_count=1)
    merged["frailty_index_fill"] = np.where(
        merged["frailty_components_available"] >= 3,
        merged["frailty_components_positive"] / merged["frailty_components_available"] * 100.0,
        np.nan,
    )
    return merged


def fill_from_long(
    core: pd.DataFrame,
    fill: pd.DataFrame,
    fill_map: dict[str, str],
    summary_rows: list[dict[str, object]],
    label: str,
) -> pd.DataFrame:
    out = core.copy()
    work = out.reset_index().merge(fill, how="left", on=["cohort", "participant_id", "wave"], validate="many_to_one")
    for target, source in fill_map.items():
        before_valid = int(out[target].notna().sum())
        values = work[source]
        fill_mask = out[target].isna().to_numpy() & values.notna().to_numpy()
        out.loc[fill_mask, target] = values.loc[fill_mask].to_numpy()
        after_valid = int(out[target].notna().sum())
        summary_rows.append(
            {
                "step": label,
                "variable": target,
                "filled_n": int(fill_mask.sum()),
                "valid_before": before_valid,
                "valid_after": after_valid,
                "valid_gain": after_valid - before_valid,
            }
        )
    return out


def variable_availability(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "age",
        "female",
        "education_group",
        "married",
        "rural",
        "fair_or_poor_srh",
        "adl_score",
        "iadl_score",
        "chronic_count",
        "depression_score",
        "depressive_symptoms",
        "cognition_z",
        "frailty_index",
        "internet_use",
        "social_participation",
        "work",
        "healthy_aging_binary",
    ]
    rows = []
    for cohort, group in df.groupby("cohort", sort=False):
        for col in key_cols:
            rows.append(
                {
                    "cohort": cohort,
                    "variable": col,
                    "valid_n": int(group[col].notna().sum()),
                    "total_n": int(len(group)),
                    "valid_pct": float(group[col].notna().mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def changed_binary_count(before: pd.Series, after: pd.Series) -> int:
    b = before.astype("Float64")
    a = after.astype("Float64")
    same = (b.eq(a)) | (b.isna() & a.isna())
    return int((~same).sum())


def write_notes(output_dir: Path, summary: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame) -> None:
    targets = ["depression_score", "depressive_symptoms", "social_participation", "frailty_index", "healthy_aging_binary"]
    lines = [
        "# Gap-filled 数据说明",
        "",
        "生成日期：2026-06-15",
        "",
        "本目录是在旧 `harmonized_core.parquet` 基础上做保守补值，原始输出未覆盖。",
        "",
        "## 补值口径",
        "",
        "- SHARE 抑郁：使用 Gateway Harmonized SHARE G 的 `r*eurod`，`depressive_symptoms = EURO-D >= 4`。",
        "- MHAS 社会参与：使用 Harmonized MHAS C.2 的 `r3socwk`-`r5socwk`，只补第 3-5 轮。",
        "- KLoSA/LASI/MHAS frailty：按可得组成项构造 0-100 代理 frailty index，至少 3 个组成项可用才给分。",
        "- Frailty 组成项包括低 BMI、疲乏/耗竭、低握力、步速/行动受限、低体力活动。KLoSA 没有标准步速变量，行动受限使用 `bedb_k` 作为保守代理。",
        "- `frail_25pct` 按 `frailty_index >= 25` 重新计算。",
        "- SHARE 抑郁补齐后，`healthy_aging_*` 派生变量已重新计算。",
        "",
        "## 补值计数",
        "",
        "| Step | Variable | Filled n | Valid before | Valid after |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['step']} | {row['variable']} | {int(row['filled_n'])} | "
            f"{int(row['valid_before'])} | {int(row['valid_after'])} |"
        )

    lines += ["", "## 目标变量覆盖率", "", "| Cohort | Variable | Before % | After % |", "|---|---|---:|---:|"]
    for cohort in after["cohort"].drop_duplicates():
        bgrp = before[before["cohort"].eq(cohort)]
        agrp = after[after["cohort"].eq(cohort)]
        for col in targets:
            b_pct = bgrp[col].notna().mean() * 100
            a_pct = agrp[col].notna().mean() * 100
            lines.append(f"| {cohort} | {col} | {b_pct:.2f} | {a_pct:.2f} |")

    lines += [
        "",
        "## 输出文件",
        "",
        "- `harmonized_core_gap_filled.parquet`：补齐后的 person-wave 核心表。",
        "- `variable_availability_gap_filled.csv`：补齐后的变量覆盖率。",
        "- `gap_fill_summary.csv`：各补值步骤的新增有效样本数。",
        "- `frailty_component_metadata.csv`：KLoSA/LASI/MHAS frailty 组成项和组件数。",
        "- `cohort_overview_gap_filled.csv`、`wave_counts_gap_filled.csv`：补齐后描述统计。",
        "",
        "## 重要限制",
        "",
        "这里的 KLoSA/LASI/MHAS frailty 是论文启动阶段的可比代理指标，不等同于各数据库已经发布的官方 frailty 成品变量。正式投稿前建议把 frailty 作为敏感性分析，或统一从所有国家重新构建同一套 deficit index。",
    ]
    (output_dir / "gap_fill_notes.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = pd.read_parquet(input_path)
    core = before.copy()
    core["participant_id"] = core["participant_id"].astype("string")
    core["wave"] = core["wave"].astype("string")

    summary_rows: list[dict[str, object]] = []

    share_depression = build_share_depression()
    core = fill_from_long(
        core,
        share_depression,
        {"depression_score": "depression_score_fill", "depressive_symptoms": "depressive_symptoms_fill"},
        summary_rows,
        "SHARE EURO-D",
    )

    mhas_social = build_mhas_social()
    core = fill_from_long(
        core,
        mhas_social,
        {"social_participation": "social_participation_fill"},
        summary_rows,
        "MHAS weekly social activity",
    )

    candidates = pd.concat(
        [build_klosa_frailty_candidates(), build_lasi_frailty_candidates(), build_mhas_frailty_candidates()],
        ignore_index=True,
    )
    frailty = add_frailty_scores(core, candidates)
    fill_mask = core["frailty_index"].isna().to_numpy() & frailty["frailty_index_fill"].notna().to_numpy()
    before_valid = int(core["frailty_index"].notna().sum())
    core.loc[fill_mask, "frailty_index"] = frailty.loc[fill_mask, "frailty_index_fill"].to_numpy()
    after_valid = int(core["frailty_index"].notna().sum())
    summary_rows.append(
        {
            "step": "KLoSA/LASI/MHAS frailty components",
            "variable": "frailty_index",
            "filled_n": int(fill_mask.sum()),
            "valid_before": before_valid,
            "valid_after": after_valid,
            "valid_gain": after_valid - before_valid,
        }
    )
    core["frail_25pct"] = np.where(core["frailty_index"].notna(), (core["frailty_index"] >= 25).astype(float), np.nan)

    core = add_derived_variables(core)
    summary_rows.append(
        {
            "step": "Recompute derived healthy ageing",
            "variable": "healthy_aging_binary",
            "filled_n": changed_binary_count(before["healthy_aging_binary"], core["healthy_aging_binary"]),
            "valid_before": int(before["healthy_aging_binary"].notna().sum()),
            "valid_after": int(core["healthy_aging_binary"].notna().sum()),
            "valid_gain": int(core["healthy_aging_binary"].notna().sum() - before["healthy_aging_binary"].notna().sum()),
        }
    )

    summary = pd.DataFrame(summary_rows)
    overview, wave_counts, _ = build_descriptives(core)
    availability = variable_availability(core)

    component_cols = [
        "cohort",
        "participant_id",
        "wave",
        "low_bmi",
        "exhaustion",
        "weakness",
        "slowness",
        "low_activity",
        "frailty_components_available",
        "frailty_components_positive",
        "frailty_index_fill",
    ]
    component_meta = frailty.loc[frailty["cohort"].isin(["KLoSA", "LASI", "MHAS"]), component_cols].copy()
    component_meta = component_meta[component_meta["frailty_index_fill"].notna()]

    core.to_parquet(output_dir / "harmonized_core_gap_filled.parquet", index=False)
    summary.to_csv(output_dir / "gap_fill_summary.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(output_dir / "variable_availability_gap_filled.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(output_dir / "cohort_overview_gap_filled.csv", index=False, encoding="utf-8-sig")
    wave_counts.to_csv(output_dir / "wave_counts_gap_filled.csv", index=False, encoding="utf-8-sig")
    component_meta.to_csv(output_dir / "frailty_component_metadata.csv", index=False, encoding="utf-8-sig")
    write_notes(output_dir, summary, before, core)

    print(f"Input rows: {len(before):,}")
    print(f"Output rows: {len(core):,}")
    print(f"Columns: {len(core.columns)}")
    print(f"Wrote outputs to {output_dir}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
