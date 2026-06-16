"""
Longitudinal analyses for the NC-oriented HRS-family ageing project.

The script uses the already harmonized and gap-filled person-wave table. It
does not read the raw cohort files. Outputs are intended as a first formal
analysis package for the "digital inclusion / social participation and healthy
ageing trajectories" manuscript idea.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_DIR / "outputs_gap_filled" / "harmonized_core_gap_filled.parquet"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs_nc_longitudinal"

CORE_COLS = [
    "cohort",
    "participant_id",
    "panel_id",
    "wave",
    "year",
    "age",
    "female",
    "education_group",
    "married",
    "rural_category",
    "work",
    "internet_use",
    "social_participation",
    "healthy_aging_binary",
    "healthy_aging_ratio",
    "frailty_index",
]

MAIN_COVARIATES = [
    "age",
    "age_sq",
    "female",
    "C(education_group)",
    "married",
    "C(rural_category)",
    "years_to_next",
    "C(wave_order)",
]


@dataclass(frozen=True)
class ModelSpec:
    exposure: str
    outcome: str = "next_healthy_aging_binary"
    baseline: str = "healthy_aging_binary"
    model_family: str = "logit"
    include_baseline: bool = True
    include_work: bool = False
    max_interval: int = 6
    min_n: int = 500


def wave_number(wave: object) -> float:
    if pd.isna(wave):
        return np.nan
    match = re.search(r"(\d+)", str(wave))
    return float(match.group(1)) if match else np.nan


def clean_for_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["wave_number"] = out["wave"].map(wave_number)
    for col in ["year", "age", "female", "married", "work", "internet_use", "social_participation"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["age_sq"] = out["age"] ** 2
    out["wave_order"] = out["wave_number"].astype("Int64").astype("string")
    out["wave_order"] = out["wave_order"].fillna(out["year"].astype("Int64").astype("string"))
    out["education_group"] = out["education_group"].astype("object").where(out["education_group"].notna(), "unknown")
    out["rural_category"] = out["rural_category"].astype("object").where(out["rural_category"].notna(), "unknown")
    out["panel_id"] = out["panel_id"].astype(str)
    out["cohort"] = out["cohort"].astype(str)
    return out.sort_values(["panel_id", "year", "wave_number"], kind="mergesort")


def build_lagged_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("panel_id", sort=False)
    for col in [
        "cohort",
        "wave",
        "year",
        "healthy_aging_binary",
        "healthy_aging_ratio",
        "frailty_index",
    ]:
        out[f"next_{col}"] = group[col].shift(-1)
    out["years_to_next"] = out["next_year"] - out["year"]
    out = out[out["cohort"].eq(out["next_cohort"])]
    out = out[out["years_to_next"].notna() & out["years_to_next"].gt(0)]
    return out


def valid_binary(series: pd.Series) -> pd.Series:
    return series.isin([0.0, 1.0, 0, 1])


def model_dataset(lagged: pd.DataFrame, spec: ModelSpec, cohorts: list[str] | None = None) -> pd.DataFrame:
    needed = [
        "panel_id",
        "cohort",
        spec.outcome,
        spec.exposure,
        spec.baseline,
        "age",
        "age_sq",
        "female",
        "education_group",
        "married",
        "rural_category",
        "work",
        "years_to_next",
        "wave_order",
    ]
    data = lagged[needed].copy()
    if cohorts is not None:
        data = data[data["cohort"].isin(cohorts)]
    data = data[data["years_to_next"].le(spec.max_interval)]
    required = [spec.outcome, spec.exposure, "age", "age_sq", "female", "married", "years_to_next", "wave_order"]
    if spec.include_work:
        required.append("work")
    data = data.dropna(subset=required)
    if spec.include_baseline:
        data = data.dropna(subset=[spec.baseline])
    data = data[valid_binary(data[spec.exposure])]
    if spec.model_family == "logit":
        data = data[valid_binary(data[spec.outcome])]
        if spec.include_baseline:
            data = data[valid_binary(data[spec.baseline])]
    for col in ["education_group", "rural_category", "cohort", "panel_id", "wave_order"]:
        data[col] = data[col].astype(str)
    return data


def formula_for(spec: ModelSpec, pooled: bool) -> str:
    rhs = [spec.exposure]
    if spec.include_baseline:
        rhs.append(spec.baseline)
    rhs.extend(MAIN_COVARIATES)
    if spec.include_work:
        rhs.append("work")
    if pooled:
        rhs.append("C(cohort)")
    return f"{spec.outcome} ~ " + " + ".join(rhs)


def fit_model(data: pd.DataFrame, spec: ModelSpec, label: str, pooled: bool) -> dict[str, object]:
    if len(data) < spec.min_n:
        return {"model": label, "exposure": spec.exposure, "n": len(data), "status": "skipped_small_n"}
    if data[spec.exposure].nunique(dropna=True) < 2:
        return {"model": label, "exposure": spec.exposure, "n": len(data), "status": "skipped_no_exposure_variation"}
    if spec.model_family == "logit" and data[spec.outcome].nunique(dropna=True) < 2:
        return {"model": label, "exposure": spec.exposure, "n": len(data), "status": "skipped_no_outcome_variation"}

    formula = formula_for(spec, pooled=pooled)
    family = sm.families.Binomial() if spec.model_family == "logit" else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spec.model_family == "logit":
                fitted = smf.glm(formula=formula, data=data, family=family).fit(
                    cov_type="cluster", cov_kwds={"groups": data["panel_id"]}
                )
            else:
                fitted = smf.ols(formula=formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["panel_id"]})
    except Exception as exc:
        return {
            "model": label,
            "exposure": spec.exposure,
            "n": len(data),
            "status": f"failed: {type(exc).__name__}: {exc}",
        }

    coef = fitted.params.get(spec.exposure, np.nan)
    se = fitted.bse.get(spec.exposure, np.nan)
    if spec.model_family == "logit":
        estimate = math.exp(coef) if pd.notna(coef) else np.nan
        low = math.exp(coef - 1.96 * se) if pd.notna(coef) and pd.notna(se) else np.nan
        high = math.exp(coef + 1.96 * se) if pd.notna(coef) and pd.notna(se) else np.nan
        estimate_name = "odds_ratio"
    else:
        estimate = coef
        low = coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan
        high = coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan
        estimate_name = "beta"
    return {
        "model": label,
        "exposure": spec.exposure,
        "outcome": spec.outcome,
        "model_family": spec.model_family,
        "include_baseline": spec.include_baseline,
        "include_work": spec.include_work,
        "max_interval": spec.max_interval,
        "n": int(fitted.nobs),
        "people": int(data["panel_id"].nunique()),
        "events": int(data[spec.outcome].sum()) if spec.model_family == "logit" else np.nan,
        "coef": coef,
        "se": se,
        estimate_name: estimate,
        "ci_low": low,
        "ci_high": high,
        "p_value": fitted.pvalues.get(spec.exposure, np.nan),
        "status": "ok",
    }


def fit_pooled_and_cohort(lagged: pd.DataFrame, spec: ModelSpec, cohorts: list[str]) -> pd.DataFrame:
    rows = []
    pooled_data = model_dataset(lagged, spec, cohorts=cohorts)
    rows.append(fit_model(pooled_data, spec, "pooled_with_cohort_wave_fixed_effects", pooled=True))
    for cohort in cohorts:
        data = model_dataset(lagged, spec, cohorts=[cohort])
        rows.append(fit_model(data, spec, f"{cohort}_specific", pooled=False))
    return pd.DataFrame(rows)


def fit_joint_exposure(lagged: pd.DataFrame, cohorts: list[str], output_dir: Path) -> pd.DataFrame:
    data = lagged[[
        "panel_id",
        "cohort",
        "next_healthy_aging_binary",
        "healthy_aging_binary",
        "internet_use",
        "social_participation",
        "age",
        "age_sq",
        "female",
        "education_group",
        "married",
        "rural_category",
        "years_to_next",
        "wave_order",
    ]].copy()
    data = data[data["cohort"].isin(cohorts)]
    data = data[data["years_to_next"].between(1, 6)]
    data = data.dropna(
        subset=[
            "next_healthy_aging_binary",
            "healthy_aging_binary",
            "internet_use",
            "social_participation",
            "age",
            "age_sq",
            "female",
            "married",
            "years_to_next",
            "wave_order",
        ]
    )
    data = data[valid_binary(data["next_healthy_aging_binary"]) & valid_binary(data["healthy_aging_binary"])]
    data = data[valid_binary(data["internet_use"]) & valid_binary(data["social_participation"])]
    data["joint_exposure"] = "neither"
    data.loc[data["internet_use"].eq(1) & data["social_participation"].eq(0), "joint_exposure"] = "internet_only"
    data.loc[data["internet_use"].eq(0) & data["social_participation"].eq(1), "joint_exposure"] = "social_only"
    data.loc[data["internet_use"].eq(1) & data["social_participation"].eq(1), "joint_exposure"] = "both"
    data["joint_exposure"] = pd.Categorical(
        data["joint_exposure"], categories=["neither", "internet_only", "social_only", "both"]
    )
    for col in ["education_group", "rural_category", "cohort", "panel_id", "wave_order"]:
        data[col] = data[col].astype(str)

    formula = (
        'next_healthy_aging_binary ~ C(joint_exposure, Treatment(reference="neither")) '
        "+ healthy_aging_binary + age + age_sq + female + C(education_group) + married + "
        "C(rural_category) + years_to_next + C(wave_order) + C(cohort)"
    )
    rows = []
    if len(data) >= 500 and data["joint_exposure"].nunique() > 1:
        try:
            fitted = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": data["panel_id"]}
            )
            covariance = "clustered_by_person"
        except Exception:
            fitted = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(cov_type="HC1")
            covariance = "HC1_fallback"
        for term, label in [
            ('C(joint_exposure, Treatment(reference="neither"))[T.internet_only]', "internet_only"),
            ('C(joint_exposure, Treatment(reference="neither"))[T.social_only]', "social_only"),
            ('C(joint_exposure, Treatment(reference="neither"))[T.both]', "both"),
        ]:
            coef = fitted.params.get(term, np.nan)
            se = fitted.bse.get(term, np.nan)
            rows.append(
                {
                    "contrast_vs_neither": label,
                    "n": int(fitted.nobs),
                    "people": int(data["panel_id"].nunique()),
                    "coef": coef,
                    "se": se,
                    "odds_ratio": math.exp(coef) if pd.notna(coef) else np.nan,
                    "ci_low": math.exp(coef - 1.96 * se) if pd.notna(coef) and pd.notna(se) else np.nan,
                    "ci_high": math.exp(coef + 1.96 * se) if pd.notna(coef) and pd.notna(se) else np.nan,
                    "p_value": fitted.pvalues.get(term, np.nan),
                    "covariance": covariance,
                    "status": "ok",
                }
            )
    counts = data.groupby(["cohort", "joint_exposure"], observed=False).size().reset_index(name="n")
    counts.to_csv(output_dir / "joint_exposure_counts.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def der_simonian_laird(cohort_rows: pd.DataFrame, exposure: str) -> dict[str, object]:
    ok = cohort_rows[(cohort_rows["status"].eq("ok")) & cohort_rows["coef"].notna() & cohort_rows["se"].gt(0)]
    if len(ok) < 2:
        return {"exposure": exposure, "k": len(ok), "status": "skipped_less_than_2_cohorts"}
    theta = ok["coef"].to_numpy(dtype=float)
    se = ok["se"].to_numpy(dtype=float)
    weights = 1 / (se**2)
    fixed = np.sum(weights * theta) / np.sum(weights)
    q = np.sum(weights * (theta - fixed) ** 2)
    df = len(theta) - 1
    c = np.sum(weights) - (np.sum(weights**2) / np.sum(weights))
    tau2 = max((q - df) / c, 0.0) if c > 0 else 0.0
    re_weights = 1 / (se**2 + tau2)
    pooled = np.sum(re_weights * theta) / np.sum(re_weights)
    pooled_se = math.sqrt(1 / np.sum(re_weights))
    i2 = max((q - df) / q, 0.0) * 100 if q > 0 else 0.0
    return {
        "exposure": exposure,
        "k": len(ok),
        "coef": pooled,
        "se": pooled_se,
        "odds_ratio": math.exp(pooled),
        "ci_low": math.exp(pooled - 1.96 * pooled_se),
        "ci_high": math.exp(pooled + 1.96 * pooled_se),
        "q": q,
        "tau2": tau2,
        "i2_pct": i2,
        "status": "ok",
    }


def transition_summary(lagged: pd.DataFrame, exposure: str, cohorts: list[str]) -> pd.DataFrame:
    data = lagged[[
        "cohort",
        "panel_id",
        exposure,
        "healthy_aging_binary",
        "next_healthy_aging_binary",
        "years_to_next",
    ]].copy()
    data = data[data["cohort"].isin(cohorts)]
    data = data[data["years_to_next"].between(1, 6)]
    data = data.dropna(subset=[exposure, "healthy_aging_binary", "next_healthy_aging_binary"])
    data = data[valid_binary(data[exposure]) & valid_binary(data["healthy_aging_binary"]) & valid_binary(data["next_healthy_aging_binary"])]
    data["exposure"] = exposure
    data["exposure_value"] = np.where(data[exposure].eq(1), "exposed", "unexposed")
    data["transition"] = np.select(
        [
            data["healthy_aging_binary"].eq(1) & data["next_healthy_aging_binary"].eq(1),
            data["healthy_aging_binary"].eq(1) & data["next_healthy_aging_binary"].eq(0),
            data["healthy_aging_binary"].eq(0) & data["next_healthy_aging_binary"].eq(1),
            data["healthy_aging_binary"].eq(0) & data["next_healthy_aging_binary"].eq(0),
        ],
        ["maintained_healthy", "declined", "improved", "remained_unhealthy"],
        default="unknown",
    )
    grouped = data.groupby(["cohort", "exposure", "exposure_value", "transition"]).size().reset_index(name="n")
    denom = grouped.groupby(["cohort", "exposure", "exposure_value"])["n"].transform("sum")
    grouped["pct"] = grouped["n"] / denom * 100.0
    return grouped


def lag_availability(lagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in lagged.groupby("cohort", sort=False):
        outcome = group[group["next_healthy_aging_binary"].notna()]
        rows.append(
            {
                "cohort": cohort,
                "lag_rows_with_next_healthy_aging": len(outcome),
                "people": outcome["panel_id"].nunique(),
                "internet_valid_rows": int(outcome["internet_use"].notna().sum()),
                "social_valid_rows": int(outcome["social_participation"].notna().sum()),
                "both_exposures_valid_rows": int(outcome[["internet_use", "social_participation"]].notna().all(axis=1).sum()),
                "median_years_to_next": float(outcome["years_to_next"].median(skipna=True)) if len(outcome) else np.nan,
                "max_years_to_next": float(outcome["years_to_next"].max(skipna=True)) if len(outcome) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sample_flow(df: pd.DataFrame, lagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in df.groupby("cohort", sort=False):
        lag = lagged[lagged["cohort"].eq(cohort)]
        rows.append(
            {
                "cohort": cohort,
                "person_waves": len(group),
                "people": group["panel_id"].nunique(),
                "waves": group["wave"].nunique(dropna=True),
                "lagged_person_waves": len(lag),
                "lagged_people": lag["panel_id"].nunique(),
                "lagged_rows_interval_1_to_6_years": int(lag["years_to_next"].between(1, 6).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_forest(model_rows: pd.DataFrame, exposure: str, meta_row: dict[str, object], output_path: Path) -> None:
    rows = model_rows[
        model_rows["model"].str.endswith("_specific", na=False) & model_rows["status"].eq("ok")
    ].copy()
    if rows.empty:
        return
    rows["label"] = rows["model"].str.replace("_specific", "", regex=False)
    rows = rows.sort_values("odds_ratio")
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, max(3, len(rows) * 0.45 + 1.5)))
    ax.errorbar(
        rows["odds_ratio"],
        y,
        xerr=[rows["odds_ratio"] - rows["ci_low"], rows["ci_high"] - rows["odds_ratio"]],
        fmt="o",
        color="#1f4e79",
        ecolor="#8aa9c4",
        capsize=3,
    )
    ax.axvline(1, color="#555555", linewidth=1, linestyle="--")
    if meta_row.get("status") == "ok":
        pooled = meta_row["odds_ratio"]
        ax.axvline(pooled, color="#b23b3b", linewidth=1.5)
        ax.text(
            pooled,
            len(rows) - 0.25,
            f"RE pooled OR {pooled:.2f}",
            color="#b23b3b",
            fontsize=9,
            ha="center",
            va="bottom",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for next-wave healthy ageing")
    ax.set_title(exposure.replace("_", " "))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_report(
    output_dir: Path,
    sample: pd.DataFrame,
    internet_models: pd.DataFrame,
    social_models: pd.DataFrame,
    meta: pd.DataFrame,
    joint: pd.DataFrame,
) -> None:
    def fmt(x: object, digits: int = 2) -> str:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"

    lines = [
        "# NC 纵向主分析第一版结果",
        "",
        "分析目标：用上一波数字包容/社会参与预测下一波健康老龄化，模拟 Nature Communications / Nature Aging 多队列 harmonized data 论文套路。",
        "",
        "## 样本流向",
        "",
        "| Cohort | Person-waves | People | Waves | Lagged rows | Lagged people | Interval 1-6y rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sample.iterrows():
        lines.append(
            f"| {row['cohort']} | {int(row['person_waves'])} | {int(row['people'])} | {int(row['waves'])} | "
            f"{int(row['lagged_person_waves'])} | {int(row['lagged_people'])} | {int(row['lagged_rows_interval_1_to_6_years'])} |"
        )

    lines += [
        "",
        "## 主模型",
        "",
        "模型：`next healthy ageing ~ exposure + baseline healthy ageing + age + age^2 + sex + education + marital status + rural category + wave + years to next wave`；pooled 辅助模型额外加入 cohort fixed effects。标准误按个体聚类。主证据采用分队列模型并做 DerSimonian-Laird random-effects meta-analysis。`work` 不放主模型，放入敏感性分析。",
        "",
        "### Internet use",
        "",
        "| Model | N | People | OR | 95% CI | P | Status |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for _, row in internet_models.iterrows():
        lines.append(
            f"| {row['model']} | {int(row.get('n', 0))} | {int(row.get('people', 0)) if pd.notna(row.get('people', np.nan)) else 0} | "
            f"{fmt(row.get('odds_ratio'))} | {fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('p_value'), 4)} | {row.get('status')} |"
        )

    lines += ["", "### Social participation", "", "| Model | N | People | OR | 95% CI | P | Status |", "|---|---:|---:|---:|---|---:|---|"]
    for _, row in social_models.iterrows():
        lines.append(
            f"| {row['model']} | {int(row.get('n', 0))} | {int(row.get('people', 0)) if pd.notna(row.get('people', np.nan)) else 0} | "
            f"{fmt(row.get('odds_ratio'))} | {fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('p_value'), 4)} | {row.get('status')} |"
        )

    lines += ["", "## Random-effects meta-analysis", "", "| Exposure | K | OR | 95% CI | I2 % | Status |", "|---|---:|---:|---|---:|---|"]
    for _, row in meta.iterrows():
        lines.append(
            f"| {row['exposure']} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} | {row.get('status')} |"
        )

    lines += ["", "## Joint exposure", "", "| Contrast vs neither | N | OR | 95% CI | P |", "|---|---:|---:|---|---:|"]
    for _, row in joint.iterrows():
        lines.append(
            f"| {row['contrast_vs_neither']} | {int(row.get('n', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('p_value'), 4)} |"
        )

    lines += [
        "",
        "## 解释边界",
        "",
        "- 这是第一版正式纵向模型，不是最终投稿版。",
        "- Internet 主分析只应解释为有标准互联网/邮件/数字使用变量的队列结果；KLoSA 和 MHAS 不强行纳入 internet 主模型。",
        "- LASI 只有一波，不进入纵向主模型，可作为横断面补充或未来新版数据补充。",
        "- 下一步应补权重、队列专属变量定义附表、失访/IPW、负对照和更严格的状态转移模型。",
    ]
    (output_dir / "analysis_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path, columns=CORE_COLS)
    df = clean_for_models(df)
    lagged = build_lagged_table(df)

    internet_cohorts = ["CHARLS", "ELSA", "HRS", "SHARE"]
    social_cohorts = ["CHARLS", "ELSA", "HRS", "KLoSA", "MHAS", "SHARE"]
    joint_cohorts = ["CHARLS", "ELSA", "HRS", "SHARE"]

    sample = sample_flow(df, lagged)
    availability = lag_availability(lagged)

    internet_main = fit_pooled_and_cohort(lagged, ModelSpec("internet_use"), internet_cohorts)
    social_main = fit_pooled_and_cohort(lagged, ModelSpec("social_participation"), social_cohorts)

    internet_sens_short = fit_pooled_and_cohort(
        lagged, ModelSpec("internet_use", max_interval=4), internet_cohorts
    ).assign(analysis="sensitivity_interval_le4")
    social_sens_short = fit_pooled_and_cohort(
        lagged, ModelSpec("social_participation", max_interval=4), social_cohorts
    ).assign(analysis="sensitivity_interval_le4")
    internet_sens_no_baseline = fit_pooled_and_cohort(
        lagged, ModelSpec("internet_use", include_baseline=False), internet_cohorts
    ).assign(analysis="sensitivity_no_baseline_healthy_ageing")
    social_sens_no_baseline = fit_pooled_and_cohort(
        lagged, ModelSpec("social_participation", include_baseline=False), social_cohorts
    ).assign(analysis="sensitivity_no_baseline_healthy_ageing")
    internet_sens_work = fit_pooled_and_cohort(
        lagged, ModelSpec("internet_use", include_work=True), internet_cohorts
    ).assign(analysis="sensitivity_add_work")
    social_sens_work = fit_pooled_and_cohort(
        lagged, ModelSpec("social_participation", include_work=True), social_cohorts
    ).assign(analysis="sensitivity_add_work")

    ratio_lagged = lagged.rename(columns={"next_healthy_aging_ratio": "next_ratio", "healthy_aging_ratio": "current_ratio"})
    ratio_lagged["next_healthy_aging_ratio"] = ratio_lagged["next_ratio"]
    ratio_lagged["healthy_aging_ratio"] = ratio_lagged["current_ratio"]
    internet_ratio = fit_pooled_and_cohort(
        ratio_lagged,
        ModelSpec("internet_use", outcome="next_healthy_aging_ratio", baseline="healthy_aging_ratio", model_family="ols"),
        internet_cohorts,
    ).assign(analysis="continuous_healthy_ageing_ratio")
    social_ratio = fit_pooled_and_cohort(
        ratio_lagged,
        ModelSpec("social_participation", outcome="next_healthy_aging_ratio", baseline="healthy_aging_ratio", model_family="ols"),
        social_cohorts,
    ).assign(analysis="continuous_healthy_ageing_ratio")

    joint = fit_joint_exposure(lagged, joint_cohorts, output_dir)

    meta_rows = [
        der_simonian_laird(internet_main[internet_main["model"].str.endswith("_specific", na=False)], "internet_use"),
        der_simonian_laird(social_main[social_main["model"].str.endswith("_specific", na=False)], "social_participation"),
    ]
    meta = pd.DataFrame(meta_rows)

    transitions = pd.concat(
        [
            transition_summary(lagged, "internet_use", internet_cohorts),
            transition_summary(lagged, "social_participation", social_cohorts),
        ],
        ignore_index=True,
    )

    sensitivity = pd.concat(
        [
            internet_sens_short,
            social_sens_short,
            internet_sens_no_baseline,
            social_sens_no_baseline,
            internet_sens_work,
            social_sens_work,
            internet_ratio,
            social_ratio,
        ],
        ignore_index=True,
    )

    sample.to_csv(output_dir / "sample_flow.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(output_dir / "lagged_availability.csv", index=False, encoding="utf-8-sig")
    internet_main.to_csv(output_dir / "model_lagged_internet_healthy_ageing.csv", index=False, encoding="utf-8-sig")
    social_main.to_csv(output_dir / "model_lagged_social_healthy_ageing.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "model_sensitivity_lagged.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(output_dir / "model_joint_internet_social.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(output_dir / "random_effects_meta_analysis.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(output_dir / "healthy_ageing_transitions_by_exposure.csv", index=False, encoding="utf-8-sig")

    plot_forest(internet_main, "internet_use", meta_rows[0], output_dir / "figure_forest_internet_use.png")
    plot_forest(social_main, "social_participation", meta_rows[1], output_dir / "figure_forest_social_participation.png")
    write_report(output_dir, sample, internet_main, social_main, meta, joint)

    print(f"Wrote NC longitudinal outputs to {output_dir}")
    print("Internet main:")
    print(internet_main.to_string(index=False))
    print("\nSocial main:")
    print(social_main.to_string(index=False))
    print("\nMeta:")
    print(meta.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
