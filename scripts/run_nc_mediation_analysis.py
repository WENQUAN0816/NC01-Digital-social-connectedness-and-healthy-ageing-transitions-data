"""
Longitudinal mediation checks for the NC connectedness manuscript.

This script rebuilds a conservative person-wave table directly from the local
Gateway/Harmonized Stata files, then estimates exploratory three-wave mediation
models:

    exposure at wave t -> mediator at wave t+1 -> healthy-ageing outcome at t+2

For each mediator, the corresponding health domain is removed from the outcome
definition to avoid circularity. The estimates are product-of-coefficients
checks on the log-odds scale, intended as mechanism evidence, not causal proof.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = Path(r"E:\nc数据\harmonized数据")
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs_nc_mediation"


@dataclass(frozen=True)
class WideCohort:
    cohort: str
    path: Path
    id_col: str
    waves: tuple[int, ...]
    year: str
    age: str
    female: str
    education: str
    married: str
    rural: str | None
    social: str | None
    internet: str | None
    srh: str | None
    adl: str | None
    iadl: str | None
    depression: str | None
    depression_threshold: float
    cognition: str | None
    chronic_suffixes: tuple[str, ...]


def cohort_specs(data_root: Path) -> list[WideCohort]:
    return [
        WideCohort(
            cohort="CHARLS",
            path=data_root / "CHARLS_Harmonized数据" / "H_CHARLS_D_Data.dta",
            id_col="ID",
            waves=(1, 2, 3, 4),
            year="r{w}iwy",
            age="r{w}agey",
            female="ragender",
            education="raeducl",
            married="r{w}mstath",
            rural="h{w}rural",
            social="r{w}socwk",
            internet=None,
            srh="r{w}shlt",
            adl="r{w}adlab_c",
            iadl="r{w}iadla",
            depression="r{w}cesd10",
            depression_threshold=10.0,
            cognition="r{w}tr20",
            chronic_suffixes=("hibpe", "diabe", "hearte", "stroke", "cancre", "lunge", "arthre"),
        ),
        WideCohort(
            cohort="ELSA",
            path=data_root / "ELSA_Harmonized数据" / "Harmonized_ELSA_G3" / "h_elsa_g3.dta",
            id_col="idauniqc",
            waves=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            year="r{w}iwindy",
            age="r{w}agey",
            female="ragender",
            education="raeducl",
            married="r{w}mstath",
            rural=None,
            social="r{w}socyr",
            internet=None,
            srh="r{w}shlt",
            adl="r{w}adltot6",
            iadl="r{w}iadlfour",
            depression="r{w}cesd",
            depression_threshold=4.0,
            cognition="r{w}tr20",
            chronic_suffixes=("hibpe", "diabe", "hearte", "stroke", "cancre", "lunge", "arthre"),
        ),
        WideCohort(
            cohort="KLoSA",
            path=data_root / "KLoSA_Harmonized数据" / "H_KLoSA_e3.dta",
            id_col="pid",
            waves=(1, 2, 3, 4, 5, 6, 7, 8),
            year="r{w}iwy",
            age="r{w}agey",
            female="ragender",
            education="raeducl",
            married="r{w}mstath",
            rural="r{w}rural",
            social="r{w}socwk",
            internet=None,
            srh="r{w}shlt",
            adl="r{w}adltotb_k",
            iadl="r{w}iadlb",
            depression="r{w}cesd10b",
            depression_threshold=10.0,
            cognition=None,
            chronic_suffixes=("hibpe", "diabe", "hearte", "stroke", "cancre", "lunge", "arthre"),
        ),
        WideCohort(
            cohort="MHAS",
            path=data_root / "MHAS_Harmonized数据" / "Harmonized MHAS (Version C.2)" / "H_MHAS_c2.dta",
            id_col="rahhidnp",
            waves=(1, 2, 3, 4, 5),
            year="r{w}iwy",
            age="r{w}agey",
            female="ragender",
            education="raeducl",
            married="r{w}mstath",
            rural="h{w}rural",
            social="r{w}socwk",
            internet=None,
            srh="r{w}shlt",
            adl="r{w}adltot6",
            iadl="r{w}iadlfour",
            depression="r{w}cesd_m",
            depression_threshold=5.0,
            cognition="r{w}tr16",
            chronic_suffixes=("hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"),
        ),
        WideCohort(
            cohort="SHARE",
            path=data_root / "SHARE_Harmonized数据" / "Gateway Harmonized SHARE G" / "GH_SHARE_g.dta",
            id_col="mergeid",
            waves=(1, 2, 4, 5, 6, 7, 8, 9),
            year="r{w}iwy",
            age="r{w}agey",
            female="ragender",
            education="raeducl",
            married="r{w}mstath",
            rural="h{w}rural",
            social="r{w}socyr",
            internet=None,
            srh="r{w}shlt",
            adl="r{w}adltot6",
            iadl="r{w}iadlfour",
            depression="r{w}eurod",
            depression_threshold=4.0,
            cognition="r{w}tr20",
            chronic_suffixes=("hibpe", "diabe", "hearte", "stroke", "cancre", "lunge", "arthre"),
        ),
    ]


def wave_col(template: str | None, wave: int) -> str | None:
    if template is None:
        return None
    return template.format(w=wave)


def existing_columns(path: Path, requested: list[str]) -> list[str]:
    labels = pd.read_stata(path, iterator=True).variable_labels()
    cols = set(labels.keys())
    return [col for col in requested if col in cols]


def id_as_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    whole = numeric.notna() & np.isfinite(numeric) & np.isclose(numeric, np.round(numeric))
    out.loc[whole] = numeric.loc[whole].round().astype("Int64").astype("string")
    return out.str.replace(r"\.0$", "", regex=True).mask(out.isna() | out.isin(["", "nan", "NaN", "<NA>"]))


def numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def binary_01(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    values = numeric(series, index=index)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[values.eq(1)] = 1.0
    out.loc[values.eq(0)] = 0.0
    return out


def married_binary(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    values = numeric(series, index=index)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[values.isin([1, 2, 3])] = 1.0
    out.loc[values.isin([4, 5, 6, 7, 8])] = 0.0
    return out


def education_group(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    values = numeric(series, index=index)
    out = pd.Series("unknown", index=values.index, dtype="object")
    out.loc[values.eq(1)] = "low"
    out.loc[values.eq(2)] = "middle"
    out.loc[values.ge(3)] = "high"
    out.loc[values.isna()] = "unknown"
    return out


def rural_category(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series("unknown", index=index, dtype="object")
    values = numeric(series, index=index)
    out = pd.Series("unknown", index=values.index, dtype="object")
    # Gateway files generally code 1 as rural and 0 as urban.
    out.loc[values.eq(1)] = "rural"
    out.loc[values.eq(0)] = "urban"
    return out


def good_srh(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    values = numeric(series, index=index)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[values.isin([1, 2, 3])] = 1.0
    out.loc[values.isin([4, 5])] = 0.0
    return out


def no_limitation(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    values = numeric(series, index=index)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[values.eq(0)] = 1.0
    out.loc[values.gt(0)] = 0.0
    return out


def not_depressive(series: pd.Series | None, threshold: float, index: pd.Index | None = None) -> pd.Series:
    values = numeric(series, index=index)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[values.lt(threshold)] = 1.0
    out.loc[values.ge(threshold)] = 0.0
    return out


def load_cohort(spec: WideCohort) -> pd.DataFrame:
    requested = [spec.id_col, spec.female, spec.education]
    for wave in spec.waves:
        for template in [
            spec.year,
            spec.age,
            spec.married,
            spec.rural,
            spec.social,
            spec.internet,
            spec.srh,
            spec.adl,
            spec.iadl,
            spec.depression,
            spec.cognition,
        ]:
            col = wave_col(template, wave)
            if col:
                requested.append(col)
        requested.extend([f"r{wave}{suffix}" for suffix in spec.chronic_suffixes])
    requested = list(dict.fromkeys(requested))
    columns = existing_columns(spec.path, requested)
    raw = pd.read_stata(spec.path, columns=columns, convert_categoricals=False)

    frames: list[pd.DataFrame] = []
    for wave in spec.waves:
        idx = raw.index
        frame = pd.DataFrame(
            {
                "cohort": spec.cohort,
                "participant_id": id_as_string(raw[spec.id_col]),
                "wave_number": float(wave),
                "wave": f"wave{wave}",
                "year": numeric(raw.get(wave_col(spec.year, wave)), idx),
                "age": numeric(raw.get(wave_col(spec.age, wave)), idx),
                "female": np.where(numeric(raw.get(spec.female), idx).eq(2), 1.0, 0.0),
                "education_group": education_group(raw.get(spec.education), idx),
                "married": married_binary(raw.get(wave_col(spec.married, wave)), idx),
                "rural_category": rural_category(raw.get(wave_col(spec.rural, wave)), idx),
                "social_participation": binary_01(raw.get(wave_col(spec.social, wave)), idx),
                "internet_use": binary_01(raw.get(wave_col(spec.internet, wave)), idx),
                "good_srh": good_srh(raw.get(wave_col(spec.srh, wave)), idx),
                "no_adl_limitation": no_limitation(raw.get(wave_col(spec.adl, wave)), idx),
                "no_iadl_limitation": no_limitation(raw.get(wave_col(spec.iadl, wave)), idx),
                "not_depressive": not_depressive(raw.get(wave_col(spec.depression, wave)), spec.depression_threshold, idx),
                "cognition_raw": numeric(raw.get(wave_col(spec.cognition, wave)), idx),
            }
        )
        disease_cols = [binary_01(raw.get(f"r{wave}{suffix}"), idx) for suffix in spec.chronic_suffixes]
        if disease_cols:
            chronic = pd.concat(disease_cols, axis=1)
            frame["chronic_count"] = chronic.sum(axis=1, min_count=1)
            frame["no_multimorbidity"] = np.where(frame["chronic_count"].notna(), (frame["chronic_count"] < 2).astype(float), np.nan)
        else:
            frame["no_multimorbidity"] = np.nan
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["participant_id"])
    out["panel_id"] = out["cohort"] + ":" + out["participant_id"].astype(str)
    out["cognition_z"] = out.groupby("cohort")["cognition_raw"].transform(
        lambda s: (s - s.mean(skipna=True)) / s.std(skipna=True)
        if s.notna().sum() > 2 and s.std(skipna=True) and not np.isclose(s.std(skipna=True), 0)
        else np.nan
    )
    out["cognition_not_low"] = np.where(out["cognition_z"].notna(), (out["cognition_z"] > -1).astype(float), np.nan)
    return out


def reduced_outcome(df: pd.DataFrame, exclude: str) -> pd.Series:
    components = {
        "srh": "good_srh",
        "function": "no_adl_limitation",
        "multimorbidity": "no_multimorbidity",
        "depression": "not_depressive",
        "cognition": "cognition_not_low",
    }
    selected = [col for key, col in components.items() if key != exclude]
    mat = df[selected]
    available = mat.notna().sum(axis=1)
    score = mat.sum(axis=1, min_count=1)
    ratio = score / available.replace(0, np.nan)
    return np.where(available.ge(3), ratio.ge(0.75).astype(float), np.nan)


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for exclude in ["cognition", "depression", "function"]:
        out[f"healthy_excl_{exclude}"] = reduced_outcome(out, exclude)
    return out


def build_triads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["panel_id", "wave_number"], kind="mergesort").copy()
    group = out.groupby("panel_id", sort=False)
    next_cols = [
        "cohort",
        "year",
        "wave_number",
        "cognition_z",
        "cognition_not_low",
        "not_depressive",
        "no_adl_limitation",
        "healthy_excl_cognition",
        "healthy_excl_depression",
        "healthy_excl_function",
    ]
    for col in next_cols:
        out[f"m_{col}"] = group[col].shift(-1)
        out[f"y_{col}"] = group[col].shift(-2)
    out["years_to_mediator"] = out["m_year"] - out["year"]
    out["years_mediator_to_outcome"] = out["y_year"] - out["m_year"]
    out = out[out["cohort"].eq(out["m_cohort"]) & out["cohort"].eq(out["y_cohort"])]
    out = out[out["years_to_mediator"].between(0.5, 8) & out["years_mediator_to_outcome"].between(0.5, 8)]
    return out


def valid_binary(series: pd.Series) -> pd.Series:
    return series.isin([0, 1, 0.0, 1.0])


def prepare_model_data(triads: pd.DataFrame, exposure: str, mediator_col: str, outcome_col: str, cohorts: list[str]) -> pd.DataFrame:
    needed = [
        "panel_id",
        "cohort",
        "wave_number",
        exposure,
        mediator_col,
        outcome_col,
        "age",
        "female",
        "education_group",
        "married",
        "rural_category",
        "years_to_mediator",
        "years_mediator_to_outcome",
    ]
    data = triads[needed].copy()
    data = data[data["cohort"].isin(cohorts)]
    data = data.dropna(subset=[exposure, mediator_col, outcome_col, "age", "female", "married"])
    data = data[valid_binary(data[exposure]) & valid_binary(data[outcome_col])]
    data["age_sq"] = data["age"] ** 2
    data["wave_order"] = data["wave_number"].astype("Int64").astype(str)
    for col in ["education_group", "rural_category", "cohort", "panel_id", "wave_order"]:
        data[col] = data[col].astype(str).fillna("unknown")
    return data


def fit_ols(formula: str, data: pd.DataFrame):
    return smf.ols(formula=formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["panel_id"]})


def fit_logit(formula: str, data: pd.DataFrame):
    return smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": data["panel_id"]}
    )


def covariate_formula(data: pd.DataFrame) -> str:
    terms = ["age", "age_sq"]
    for col in ["female", "married", "years_to_mediator", "years_mediator_to_outcome"]:
        if data[col].nunique(dropna=True) > 1:
            terms.append(col)
    for col in ["education_group", "rural_category", "wave_order", "cohort"]:
        if data[col].nunique(dropna=True) > 1:
            terms.append(f"C({col})")
    return " + ".join(terms)


def mediation_one(data: pd.DataFrame, exposure: str, mediator: str, mediator_col: str, outcome_col: str) -> dict[str, object]:
    if len(data) < 500:
        return {"status": "skipped_small_n", "n": len(data)}
    if data[exposure].nunique(dropna=True) < 2 or data[outcome_col].nunique(dropna=True) < 2:
        return {"status": "skipped_no_variation", "n": len(data)}

    covars = covariate_formula(data)
    try:
        a_model = fit_ols(f"{mediator_col} ~ {exposure} + {covars}", data)
        total_model = fit_logit(f"{outcome_col} ~ {exposure} + {covars}", data)
        direct_model = fit_logit(f"{outcome_col} ~ {exposure} + {mediator_col} + {covars}", data)
    except Exception as exc:
        return {"status": f"failed: {type(exc).__name__}: {exc}", "n": len(data)}

    a = float(a_model.params.get(exposure, np.nan))
    a_se = float(a_model.bse.get(exposure, np.nan))
    b = float(direct_model.params.get(mediator_col, np.nan))
    b_se = float(direct_model.bse.get(mediator_col, np.nan))
    c = float(total_model.params.get(exposure, np.nan))
    c_se = float(total_model.bse.get(exposure, np.nan))
    cprime = float(direct_model.params.get(exposure, np.nan))
    cprime_se = float(direct_model.bse.get(exposure, np.nan))
    indirect = a * b
    indirect_se = math.sqrt((b * b * a_se * a_se) + (a * a * b_se * b_se)) if np.isfinite([a, b, a_se, b_se]).all() else np.nan
    prop = indirect / c * 100 if c and np.isfinite(c) else np.nan

    return {
        "status": "ok",
        "n": int(len(data)),
        "people": int(data["panel_id"].nunique()),
        "cohorts": ",".join(sorted(data["cohort"].unique())),
        "exposure_prevalence_pct": float(data[exposure].mean() * 100),
        "outcome_prevalence_pct": float(data[outcome_col].mean() * 100),
        "a_coef": a,
        "a_se": a_se,
        "a_p": float(a_model.pvalues.get(exposure, np.nan)),
        "b_coef": b,
        "b_se": b_se,
        "b_or": math.exp(b) if np.isfinite(b) else np.nan,
        "b_p": float(direct_model.pvalues.get(mediator_col, np.nan)),
        "total_coef": c,
        "total_se": c_se,
        "total_or": math.exp(c) if np.isfinite(c) else np.nan,
        "total_ci_low": math.exp(c - 1.96 * c_se) if np.isfinite([c, c_se]).all() else np.nan,
        "total_ci_high": math.exp(c + 1.96 * c_se) if np.isfinite([c, c_se]).all() else np.nan,
        "total_p": float(total_model.pvalues.get(exposure, np.nan)),
        "direct_coef": cprime,
        "direct_se": cprime_se,
        "direct_or": math.exp(cprime) if np.isfinite(cprime) else np.nan,
        "direct_ci_low": math.exp(cprime - 1.96 * cprime_se) if np.isfinite([cprime, cprime_se]).all() else np.nan,
        "direct_ci_high": math.exp(cprime + 1.96 * cprime_se) if np.isfinite([cprime, cprime_se]).all() else np.nan,
        "direct_p": float(direct_model.pvalues.get(exposure, np.nan)),
        "indirect_coef": indirect,
        "indirect_se": indirect_se,
        "indirect_ci_low": indirect - 1.96 * indirect_se if np.isfinite(indirect_se) else np.nan,
        "indirect_ci_high": indirect + 1.96 * indirect_se if np.isfinite(indirect_se) else np.nan,
        "indirect_or": math.exp(indirect) if np.isfinite(indirect) else np.nan,
        "proportion_mediated_pct": prop,
    }


def run(data_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    availability_rows = []
    for spec in cohort_specs(data_root):
        if not spec.path.exists():
            availability_rows.append({"cohort": spec.cohort, "status": f"missing_file: {spec.path}"})
            continue
        frame = load_cohort(spec)
        frames.append(frame)
        availability_rows.append(
            {
                "cohort": spec.cohort,
                "status": "loaded",
                "person_waves": len(frame),
                "people": frame["panel_id"].nunique(),
                "waves": frame["wave_number"].nunique(),
                "social_valid": int(frame["social_participation"].notna().sum()),
                "internet_valid": int(frame["internet_use"].notna().sum()),
                "cognition_valid": int(frame["cognition_z"].notna().sum()),
                "depression_valid": int(frame["not_depressive"].notna().sum()),
                "function_valid": int(frame["no_adl_limitation"].notna().sum()),
            }
        )

    if not frames:
        raise RuntimeError("No cohort data loaded.")

    person_wave = add_outcomes(pd.concat(frames, ignore_index=True))
    triads = build_triads(person_wave)

    analyses = [
        {
            "analysis": "social_to_cognition",
            "exposure": "social_participation",
            "mediator": "cognition_z",
            "mediator_col": "m_cognition_z",
            "outcome": "healthy_excl_cognition",
            "outcome_col": "y_healthy_excl_cognition",
        },
        {
            "analysis": "social_to_not_depressive",
            "exposure": "social_participation",
            "mediator": "not_depressive",
            "mediator_col": "m_not_depressive",
            "outcome": "healthy_excl_depression",
            "outcome_col": "y_healthy_excl_depression",
        },
        {
            "analysis": "social_to_no_adl_limitation",
            "exposure": "social_participation",
            "mediator": "no_adl_limitation",
            "mediator_col": "m_no_adl_limitation",
            "outcome": "healthy_excl_function",
            "outcome_col": "y_healthy_excl_function",
        },
    ]

    rows = []
    cohort_rows = []
    for item in analyses:
        data = prepare_model_data(
            triads,
            exposure=item["exposure"],
            mediator_col=item["mediator_col"],
            outcome_col=item["outcome_col"],
            cohorts=sorted(person_wave["cohort"].unique()),
        )
        result = mediation_one(data, item["exposure"], item["mediator"], item["mediator_col"], item["outcome_col"])
        rows.append({**item, "scope": "pooled", **result})
        for cohort in sorted(data["cohort"].dropna().unique()):
            cohort_data = data[data["cohort"].eq(cohort)].copy()
            cohort_result = mediation_one(
                cohort_data,
                item["exposure"],
                item["mediator"],
                item["mediator_col"],
                item["outcome_col"],
            )
            cohort_rows.append({**item, "scope": cohort, **cohort_result})

    availability = pd.DataFrame(availability_rows)
    summary = pd.DataFrame(rows)
    by_cohort = pd.DataFrame(cohort_rows)
    triad_availability = (
        triads.groupby("cohort")
        .agg(triads=("panel_id", "size"), people=("panel_id", "nunique"), waves=("wave_number", "nunique"))
        .reset_index()
    )

    availability.to_csv(output_dir / "mediation_variable_availability.csv", index=False, encoding="utf-8-sig")
    triad_availability.to_csv(output_dir / "mediation_triad_availability.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "mediation_summary.csv", index=False, encoding="utf-8-sig")
    by_cohort.to_csv(output_dir / "mediation_by_cohort.csv", index=False, encoding="utf-8-sig")

    write_report(output_dir, availability, triad_availability, summary, by_cohort)
    print(f"Wrote mediation outputs to {output_dir}")
    print(summary.to_string(index=False))


def fmt(x: object, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def int_fmt(x: object) -> str:
    try:
        if pd.isna(x):
            return "0"
        return str(int(float(x)))
    except Exception:
        return "0"


def write_report(
    output_dir: Path,
    availability: pd.DataFrame,
    triads: pd.DataFrame,
    summary: pd.DataFrame,
    by_cohort: pd.DataFrame,
) -> None:
    lines = [
        "# NC01 纵向中介分析第一版",
        "",
        "设计：`wave t` 的 connectedness 暴露 -> `wave t+1` 的中介 -> `wave t+2` 的健康老龄化结局。",
        "",
        "为避免循环定义，检验某一中介时，结局删除对应健康域：认知中介使用 `healthy_excl_cognition`，抑郁中介使用 `healthy_excl_depression`，功能中介使用 `healthy_excl_function`。",
        "",
        "模型为探索性 product-of-coefficients：中介模型用线性模型，结局模型用 logistic GLM，标准误按个体聚类。协变量包括年龄、年龄平方、性别、教育、婚姻、城乡、两段随访间隔、基线波次和队列固定效应。",
        "",
        "## 队列可用性",
        "",
        "| Cohort | Person-waves | People | Social valid | Internet valid | Cognition valid | Depression valid | Function valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in availability.iterrows():
        lines.append(
            f"| {row.get('cohort')} | {int_fmt(row.get('person_waves'))} | {int_fmt(row.get('people'))} | "
            f"{int_fmt(row.get('social_valid'))} | {int_fmt(row.get('internet_valid'))} | "
            f"{int_fmt(row.get('cognition_valid'))} | {int_fmt(row.get('depression_valid'))} | "
            f"{int_fmt(row.get('function_valid'))} |"
        )

    lines += [
        "",
        "## 三波样本",
        "",
        "| Cohort | Triad rows | People | Waves |",
        "|---|---:|---:|---:|",
    ]
    for _, row in triads.iterrows():
        lines.append(f"| {row['cohort']} | {int(row['triads'])} | {int(row['people'])} | {int(row['waves'])} |")

    lines += [
        "",
        "## 中介结果",
        "",
        "| Analysis | N | People | Cohorts | Total OR | Direct OR | Indirect OR | Indirect log-OR 95% CI | Mediated % | Status |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['analysis']} | {int_fmt(row.get('n'))} | {int_fmt(row.get('people'))} | "
            f"{row.get('cohorts', '')} | {fmt(row.get('total_or'))} | {fmt(row.get('direct_or'))} | "
            f"{fmt(row.get('indirect_or'))} | {fmt(row.get('indirect_ci_low'), 4)} to {fmt(row.get('indirect_ci_high'), 4)} | "
            f"{fmt(row.get('proportion_mediated_pct'))} | {row.get('status')} |"
        )

    ok_cohort = by_cohort[by_cohort["status"].eq("ok")].copy() if "status" in by_cohort else pd.DataFrame()
    if not ok_cohort.empty:
        lines += [
            "",
            "## 队列特异性结果",
            "",
            "| Analysis | Cohort | N | Total OR | Direct OR | Indirect OR | Mediated % |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in ok_cohort.iterrows():
            lines.append(
                f"| {row['analysis']} | {row['scope']} | {int_fmt(row.get('n'))} | "
                f"{fmt(row.get('total_or'))} | {fmt(row.get('direct_or'))} | "
                f"{fmt(row.get('indirect_or'))} | {fmt(row.get('proportion_mediated_pct'))} |"
            )

    lines += [
        "",
        "## 解释边界",
        "",
        "- 这是机制增强的第一版结果，适合判断是否值得加入论文，不应直接改写为因果机制证明。",
        "- 本机 `E:\\nc数据` 的 Stata 宽表中，CHARLS/ELSA/SHARE 的 internet 变量未能用当前文件可靠定位；因此本版 mediation 输出只报告社会参与路径，internet mediation 不作为多队列主结果。",
        "- 社会参与路径覆盖现有可定位队列；若后续找到原始 `01_data_deduped/csv` 或生成 `harmonized_core_gap_filled.parquet`，可以按同一脚本补齐 internet/joint mediation。",
    ]
    (output_dir / "mediation_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
