"""
NC-level upgrade analyses for the harmonized ageing cohort project.

This script builds an expanded evidence package around connectedness and
healthy ageing trajectories:
- cohort-specific joint exposure models and random-effects meta-analysis
- transition models for maintaining or gaining healthy ageing
- domain outcome models for the healthy ageing components
- subgroup heterogeneity models
- leave-one-cohort-out meta-analysis
- attrition/IPW sensitivity
- alternative healthy ageing definitions
- E-values for main meta estimates

It reads the gap-filled harmonized parquet only and writes a separate output
directory, leaving raw data and first-round outputs unchanged.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_DIR / "outputs_gap_filled" / "harmonized_core_gap_filled.parquet"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs_nc_upgrade"
LONGITUDINAL_OUTPUT_DIR = PIPELINE_DIR / "outputs_nc_longitudinal"

INTERNET_COHORTS = ["CHARLS", "ELSA", "HRS", "SHARE"]
SOCIAL_COHORTS = ["CHARLS", "ELSA", "HRS", "KLoSA", "MHAS", "SHARE"]
JOINT_COHORTS = ["CHARLS", "ELSA", "HRS", "SHARE"]

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
    "fair_or_poor_srh",
    "adl_limitation",
    "iadl_limitation",
    "multimorbidity",
    "depressive_symptoms",
    "cognition_z",
    "frailty_index",
    "internet_use",
    "social_participation",
    "work",
    "healthy_aging_components",
    "healthy_aging_score",
    "healthy_aging_ratio",
    "healthy_aging_binary",
]

DOMAIN_OUTCOMES = {
    "good_self_rated_health": "good_srh",
    "no_adl_limitation": "no_adl_limitation",
    "no_multimorbidity": "no_multimorbidity",
    "not_depressive": "not_depressive",
    "cognition_not_low": "cognition_not_low",
}

BASE_COVARIATES = [
    ("age", "age"),
    ("age_sq", "age_sq"),
    ("female", "female"),
    ("education_group", "C(education_group)"),
    ("married", "married"),
    ("rural_category", "C(rural_category)"),
    ("years_to_next", "years_to_next"),
    ("wave_order", "C(wave_order)"),
]

JOINT_TERMS = [
    ('C(joint_exposure, Treatment(reference="neither"))[T.internet_only]', "internet_only"),
    ('C(joint_exposure, Treatment(reference="neither"))[T.social_only]', "social_only"),
    ('C(joint_exposure, Treatment(reference="neither"))[T.both]', "both"),
]


def wave_number(wave: object) -> float:
    if pd.isna(wave):
        return np.nan
    match = re.search(r"(\d+)", str(wave))
    return float(match.group(1)) if match else np.nan


def valid_binary(series: pd.Series) -> pd.Series:
    return series.isin([0.0, 1.0, 0, 1])


def safe_exp(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return float(math.exp(float(np.clip(x, -50, 50))))


def clean_for_upgrade(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_id"] = np.arange(len(out), dtype=np.int64)
    out["wave_number"] = out["wave"].map(wave_number)
    numeric_cols = [
        "year",
        "age",
        "female",
        "married",
        "work",
        "fair_or_poor_srh",
        "adl_limitation",
        "iadl_limitation",
        "multimorbidity",
        "depressive_symptoms",
        "cognition_z",
        "frailty_index",
        "internet_use",
        "social_participation",
        "healthy_aging_components",
        "healthy_aging_score",
        "healthy_aging_ratio",
        "healthy_aging_binary",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["age_sq"] = out["age"] ** 2
    out["wave_order"] = out["wave_number"].astype("Int64").astype("string")
    out["wave_order"] = out["wave_order"].fillna(out["year"].astype("Int64").astype("string"))
    out["education_group"] = out["education_group"].astype("object").where(out["education_group"].notna(), "unknown")
    out["rural_category"] = out["rural_category"].astype("object").where(out["rural_category"].notna(), "unknown")
    out["panel_id"] = out["panel_id"].astype(str)
    out["cohort"] = out["cohort"].astype(str)

    out["good_srh"] = np.where(out["fair_or_poor_srh"].notna(), 1 - out["fair_or_poor_srh"], np.nan)
    out["no_adl_limitation"] = np.where(out["adl_limitation"].notna(), 1 - out["adl_limitation"], np.nan)
    out["no_iadl_limitation"] = np.where(out["iadl_limitation"].notna(), 1 - out["iadl_limitation"], np.nan)
    out["no_multimorbidity"] = np.where(out["multimorbidity"].notna(), 1 - out["multimorbidity"], np.nan)
    out["not_depressive"] = np.where(out["depressive_symptoms"].notna(), 1 - out["depressive_symptoms"], np.nan)
    out["cognition_not_low"] = np.where(out["cognition_z"].notna(), (out["cognition_z"] > -1).astype(float), np.nan)

    out["healthy_strict_binary"] = np.where(
        out["healthy_aging_components"].ge(5),
        out["healthy_aging_ratio"].ge(1.0).astype(float),
        np.nan,
    )
    out["healthy_lenient_binary"] = np.where(
        out["healthy_aging_components"].ge(4),
        out["healthy_aging_ratio"].ge(0.6).astype(float),
        np.nan,
    )
    reduced_components = out[["good_srh", "no_adl_limitation", "no_multimorbidity", "not_depressive"]]
    reduced_n = reduced_components.notna().sum(axis=1)
    reduced_ratio = reduced_components.sum(axis=1, min_count=1) / reduced_n.replace(0, np.nan)
    out["healthy_no_cognition_binary"] = np.where(reduced_n.ge(3), reduced_ratio.ge(0.75).astype(float), np.nan)

    return out.sort_values(["panel_id", "year", "wave_number"], kind="mergesort")


def build_lagged_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("panel_id", sort=False)
    for col in ["cohort", "year", "internet_use", "social_participation"]:
        out[f"prev_{col}"] = group[col].shift(1)
    out["years_from_prev"] = out["year"] - out["prev_year"]

    lag_cols = [
        "cohort",
        "wave",
        "year",
        "healthy_aging_binary",
        "healthy_aging_ratio",
        "healthy_strict_binary",
        "healthy_lenient_binary",
        "healthy_no_cognition_binary",
        "frailty_index",
        "good_srh",
        "no_adl_limitation",
        "no_multimorbidity",
        "not_depressive",
        "cognition_not_low",
    ]
    for col in lag_cols:
        out[f"next_{col}"] = group[col].shift(-1)
    out["years_to_next"] = out["next_year"] - out["year"]
    out = out[out["cohort"].eq(out["next_cohort"])]
    out = out[out["years_to_next"].notna() & out["years_to_next"].gt(0)]
    return out


def add_joint_exposure(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["joint_exposure"] = "neither"
    out.loc[out["internet_use"].eq(1) & out["social_participation"].eq(0), "joint_exposure"] = "internet_only"
    out.loc[out["internet_use"].eq(0) & out["social_participation"].eq(1), "joint_exposure"] = "social_only"
    out.loc[out["internet_use"].eq(1) & out["social_participation"].eq(1), "joint_exposure"] = "both"
    out["joint_exposure"] = pd.Categorical(
        out["joint_exposure"], categories=["neither", "internet_only", "social_only", "both"]
    )
    return out


def covariate_terms(skip_raw: set[str] | None = None) -> list[str]:
    skip_raw = skip_raw or set()
    return [term for raw, term in BASE_COVARIATES if raw not in skip_raw]


def model_frame(
    lagged: pd.DataFrame,
    cohorts: list[str],
    exposure_cols: list[str],
    outcome: str,
    baseline: str | None = None,
    max_interval: int = 6,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    cols = [
        "row_id",
        "panel_id",
        "cohort",
        outcome,
        "age",
        "age_sq",
        "female",
        "education_group",
        "married",
        "rural_category",
        "years_to_next",
        "wave_order",
    ]
    cols.extend(exposure_cols)
    if baseline is not None:
        cols.append(baseline)
    if extra_cols:
        cols.extend(extra_cols)
    cols = list(dict.fromkeys(cols))
    data = lagged[cols].copy()
    data = data[data["cohort"].isin(cohorts)]
    data = data[data["years_to_next"].between(1, max_interval)]
    required = [
        outcome,
        "age",
        "age_sq",
        "female",
        "married",
        "years_to_next",
        "wave_order",
        *exposure_cols,
    ]
    if baseline is not None:
        required.append(baseline)
    data = data.dropna(subset=required)
    data = data[valid_binary(data[outcome])]
    for col in exposure_cols:
        if col in ["internet_use", "social_participation"]:
            data = data[valid_binary(data[col])]
    if baseline is not None:
        data = data[valid_binary(data[baseline])]
    for col in ["education_group", "rural_category", "cohort", "panel_id", "wave_order"]:
        data[col] = data[col].astype(str)
    return data


def fit_terms(
    data: pd.DataFrame,
    formula: str,
    terms: list[tuple[str, str]],
    metadata: dict[str, object],
    weight_col: str | None = None,
    min_n: int = 500,
) -> list[dict[str, object]]:
    base = dict(metadata)
    base["n"] = int(len(data))
    base["people"] = int(data["panel_id"].nunique()) if "panel_id" in data.columns else np.nan
    if len(data) < min_n:
        return [{**base, "term": label, "term_raw": term, "status": "skipped_small_n"} for term, label in terms]

    outcome = formula.split("~", 1)[0].strip()
    if outcome in data.columns and data[outcome].nunique(dropna=True) < 2:
        return [{**base, "term": label, "term_raw": term, "status": "skipped_no_outcome_variation"} for term, label in terms]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if weight_col is None:
                model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
            else:
                model = smf.glm(
                    formula=formula,
                    data=data,
                    family=sm.families.Binomial(),
                    freq_weights=data[weight_col],
                )
            try:
                fitted = model.fit(cov_type="cluster", cov_kwds={"groups": data["panel_id"]})
                covariance = "clustered_by_person"
            except Exception:
                fitted = model.fit(cov_type="HC1")
                covariance = "HC1_fallback"
    except Exception as exc:
        return [
            {**base, "term": label, "term_raw": term, "status": f"failed: {type(exc).__name__}: {exc}"}
            for term, label in terms
        ]

    rows = []
    for term, label in terms:
        coef = fitted.params.get(term, np.nan)
        se = fitted.bse.get(term, np.nan)
        if pd.isna(coef) or pd.isna(se):
            rows.append({**base, "term": label, "term_raw": term, "status": "term_absent"})
            continue
        rows.append(
            {
                **base,
                "term": label,
                "term_raw": term,
                "n": int(fitted.nobs),
                "people": int(data["panel_id"].nunique()),
                "events": int(data[outcome].sum()) if outcome in data.columns else np.nan,
                "coef": float(coef),
                "se": float(se),
                "odds_ratio": safe_exp(coef),
                "ci_low": safe_exp(coef - 1.96 * se),
                "ci_high": safe_exp(coef + 1.96 * se),
                "p_value": float(fitted.pvalues.get(term, np.nan)),
                "covariance": covariance,
                "weighted": bool(weight_col),
                "status": "ok",
            }
        )
    return rows


def single_exposure_formula(
    outcome: str,
    exposure: str,
    baseline: str | None,
    pooled: bool = False,
    skip_covariates: set[str] | None = None,
) -> str:
    rhs = [exposure]
    if baseline is not None:
        rhs.append(baseline)
    rhs.extend(covariate_terms(skip_covariates))
    if pooled:
        rhs.append("C(cohort)")
    return f"{outcome} ~ " + " + ".join(rhs)


def joint_formula(
    outcome: str,
    baseline: str | None,
    pooled: bool = False,
    skip_covariates: set[str] | None = None,
) -> str:
    rhs = ['C(joint_exposure, Treatment(reference="neither"))']
    if baseline is not None:
        rhs.append(baseline)
    rhs.extend(covariate_terms(skip_covariates))
    if pooled:
        rhs.append("C(cohort)")
    return f"{outcome} ~ " + " + ".join(rhs)


def random_effects_meta(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out_rows = []
    if rows.empty:
        return pd.DataFrame()
    for keys, group in rows.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(by, keys))
        ok = group[group["status"].eq("ok") & group["coef"].notna() & group["se"].gt(0)].copy()
        meta["k"] = int(len(ok))
        if len(ok) < 2:
            out_rows.append({**meta, "status": "skipped_less_than_2_cohorts"})
            continue
        theta = ok["coef"].to_numpy(dtype=float)
        se = ok["se"].to_numpy(dtype=float)
        weights = 1 / (se**2)
        fixed = np.sum(weights * theta) / np.sum(weights)
        q = np.sum(weights * (theta - fixed) ** 2)
        df = len(theta) - 1
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau2 = max((q - df) / c, 0.0) if c > 0 else 0.0
        re_weights = 1 / (se**2 + tau2)
        pooled = np.sum(re_weights * theta) / np.sum(re_weights)
        pooled_se = math.sqrt(1 / np.sum(re_weights))
        i2 = max((q - df) / q, 0.0) * 100 if q > 0 else 0.0
        out_rows.append(
            {
                **meta,
                "coef": float(pooled),
                "se": float(pooled_se),
                "odds_ratio": safe_exp(pooled),
                "ci_low": safe_exp(pooled - 1.96 * pooled_se),
                "ci_high": safe_exp(pooled + 1.96 * pooled_se),
                "q": float(q),
                "tau2": float(tau2),
                "i2_pct": float(i2),
                "status": "ok",
            }
        )
    return pd.DataFrame(out_rows)


def random_effects_continuous(rows: pd.DataFrame, by: list[str], estimate_col: str, se_col: str) -> pd.DataFrame:
    out_rows = []
    if rows.empty:
        return pd.DataFrame()
    for keys, group in rows.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(by, keys))
        ok = group[group["status"].eq("ok") & group[estimate_col].notna() & group[se_col].gt(0)].copy()
        meta["k"] = int(len(ok))
        if len(ok) < 2:
            out_rows.append({**meta, "status": "skipped_less_than_2_cohorts"})
            continue
        theta = ok[estimate_col].to_numpy(dtype=float)
        se = ok[se_col].to_numpy(dtype=float)
        weights = 1 / (se**2)
        fixed = np.sum(weights * theta) / np.sum(weights)
        q = np.sum(weights * (theta - fixed) ** 2)
        df = len(theta) - 1
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau2 = max((q - df) / c, 0.0) if c > 0 else 0.0
        re_weights = 1 / (se**2 + tau2)
        pooled = np.sum(re_weights * theta) / np.sum(re_weights)
        pooled_se = math.sqrt(1 / np.sum(re_weights))
        i2 = max((q - df) / q, 0.0) * 100 if q > 0 else 0.0
        out_rows.append(
            {
                **meta,
                estimate_col: float(pooled),
                se_col: float(pooled_se),
                "ci_low": float(pooled - 1.96 * pooled_se),
                "ci_high": float(pooled + 1.96 * pooled_se),
                "q": float(q),
                "tau2": float(tau2),
                "i2_pct": float(i2),
                "status": "ok",
            }
        )
    return pd.DataFrame(out_rows)


def e_value(or_value: float) -> float:
    if pd.isna(or_value):
        return np.nan
    rr = float(or_value)
    if rr < 1:
        rr = 1 / rr
    if rr <= 1:
        return 1.0
    return rr + math.sqrt(rr * (rr - 1))


def build_e_values(meta: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in meta.iterrows():
        if row.get("status") != "ok":
            continue
        lower = row.get("ci_low", np.nan)
        labels = {}
        for col in label_cols:
            value = row.get(col)
            if pd.isna(value) and col == "term":
                value = row.get("exposure")
            labels[col] = value
        rows.append(
            {
                **labels,
                "odds_ratio": row.get("odds_ratio"),
                "ci_low": lower,
                "ci_high": row.get("ci_high"),
                "e_value_point": e_value(row.get("odds_ratio", np.nan)),
                "e_value_ci": e_value(lower) if pd.notna(lower) and float(lower) > 1 else 1.0,
            }
        )
    return pd.DataFrame(rows)


def run_joint_by_cohort(lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cohort in JOINT_COHORTS:
        data = model_frame(
            lagged,
            [cohort],
            ["internet_use", "social_participation"],
            "next_healthy_aging_binary",
            baseline="healthy_aging_binary",
        )
        data = add_joint_exposure(data)
        formula = joint_formula("next_healthy_aging_binary", "healthy_aging_binary", pooled=False)
        rows.extend(
            fit_terms(
                data,
                formula,
                JOINT_TERMS,
                {"analysis": "joint_exposure", "cohort": cohort, "outcome": "next_healthy_aging_binary"},
            )
        )
    model_rows = pd.DataFrame(rows)
    meta = random_effects_meta(model_rows, ["analysis", "outcome", "term"])
    return model_rows, meta


def run_joint_interaction_models(lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for cohort in JOINT_COHORTS:
        data = model_frame(
            lagged,
            [cohort],
            ["internet_use", "social_participation"],
            "next_healthy_aging_binary",
            baseline="healthy_aging_binary",
        )
        data["internet_social"] = data["internet_use"] * data["social_participation"]
        formula = (
            "next_healthy_aging_binary ~ internet_use + social_participation + internet_social + "
            "healthy_aging_binary + " + " + ".join(covariate_terms())
        )
        base = {
            "analysis": "joint_interaction",
            "cohort": cohort,
            "outcome": "next_healthy_aging_binary",
            "term": "internet_x_social_multiplicative",
            "n": int(len(data)),
            "people": int(data["panel_id"].nunique()) if len(data) else 0,
        }
        if len(data) < 500:
            rows.append({**base, "status": "skipped_small_n"})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
                try:
                    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": data["panel_id"]})
                    covariance = "clustered_by_person"
                except Exception:
                    fitted = model.fit(cov_type="HC1")
                    covariance = "HC1_fallback"
            terms = ["internet_use", "social_participation", "internet_social"]
            if any(term not in fitted.params.index for term in terms):
                rows.append({**base, "status": "term_absent"})
                continue
            b = fitted.params[terms].to_numpy(dtype=float)
            cov = fitted.cov_params().loc[terms, terms].to_numpy(dtype=float)
            b1, b2, b3 = b
            or10 = safe_exp(b1)
            or01 = safe_exp(b2)
            or11 = safe_exp(b1 + b2 + b3)
            reri = or11 - or10 - or01 + 1
            gradient = np.array([or11 - or10, or11 - or01, or11], dtype=float)
            reri_var = float(gradient @ cov @ gradient.T)
            reri_se = math.sqrt(max(reri_var, 0.0)) if np.isfinite(reri_var) else np.nan
            reri_p = math.erfc(abs(reri / reri_se) / math.sqrt(2)) if pd.notna(reri_se) and reri_se > 0 else np.nan
            ap = reri / or11 if or11 else np.nan
            denom = (or10 - 1) + (or01 - 1)
            synergy_index = (or11 - 1) / denom if denom > 0 else np.nan
            interaction_se = float(fitted.bse["internet_social"])
            rows.append(
                {
                    **base,
                    "events": int(data["next_healthy_aging_binary"].sum()),
                    "coef": float(b3),
                    "se": interaction_se,
                    "odds_ratio": safe_exp(b3),
                    "ci_low": safe_exp(b3 - 1.96 * interaction_se),
                    "ci_high": safe_exp(b3 + 1.96 * interaction_se),
                    "p_value": float(fitted.pvalues.get("internet_social", np.nan)),
                    "or_internet_only": or10,
                    "or_social_only": or01,
                    "or_both": or11,
                    "reri": float(reri),
                    "reri_se": float(reri_se) if pd.notna(reri_se) else np.nan,
                    "reri_ci_low": float(reri - 1.96 * reri_se) if pd.notna(reri_se) else np.nan,
                    "reri_ci_high": float(reri + 1.96 * reri_se) if pd.notna(reri_se) else np.nan,
                    "reri_p_value": float(reri_p) if pd.notna(reri_p) else np.nan,
                    "attributable_proportion": float(ap) if pd.notna(ap) else np.nan,
                    "synergy_index": float(synergy_index) if pd.notna(synergy_index) else np.nan,
                    "covariance": covariance,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({**base, "status": f"failed: {type(exc).__name__}: {exc}"})
    model_rows = pd.DataFrame(rows)
    mult_meta = random_effects_meta(model_rows, ["analysis", "outcome", "term"])
    reri_meta = random_effects_continuous(model_rows, ["analysis", "outcome"], "reri", "reri_se")
    return model_rows, mult_meta, reri_meta


def run_exposure_history_models(lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    terms = [
        ('C(exposure_history, Treatment(reference="never"))[T.adopted]', "adopted"),
        ('C(exposure_history, Treatment(reference="never"))[T.stopped]', "stopped"),
        ('C(exposure_history, Treatment(reference="never"))[T.persistent]', "persistent"),
    ]
    for exposure, cohorts in [("internet_use", INTERNET_COHORTS), ("social_participation", SOCIAL_COHORTS)]:
        prev_exposure = f"prev_{exposure}"
        for cohort in cohorts:
            cols = [
                "panel_id",
                "cohort",
                "prev_cohort",
                "next_healthy_aging_binary",
                "healthy_aging_binary",
                exposure,
                prev_exposure,
                "age",
                "age_sq",
                "female",
                "education_group",
                "married",
                "rural_category",
                "years_from_prev",
                "years_to_next",
                "wave_order",
            ]
            data = lagged[cols].copy()
            data = data[data["cohort"].eq(cohort)]
            data = data[data["prev_cohort"].eq(data["cohort"])]
            data = data[data["years_from_prev"].between(1, 6) & data["years_to_next"].between(1, 6)]
            data = data.dropna(
                subset=[
                    "next_healthy_aging_binary",
                    "healthy_aging_binary",
                    exposure,
                    prev_exposure,
                    "age",
                    "age_sq",
                    "female",
                    "married",
                    "years_to_next",
                    "wave_order",
                ]
            )
            data = data[
                valid_binary(data["next_healthy_aging_binary"])
                & valid_binary(data["healthy_aging_binary"])
                & valid_binary(data[exposure])
                & valid_binary(data[prev_exposure])
            ]
            data["exposure_history"] = "never"
            data.loc[data[prev_exposure].eq(0) & data[exposure].eq(1), "exposure_history"] = "adopted"
            data.loc[data[prev_exposure].eq(1) & data[exposure].eq(0), "exposure_history"] = "stopped"
            data.loc[data[prev_exposure].eq(1) & data[exposure].eq(1), "exposure_history"] = "persistent"
            data["exposure_history"] = pd.Categorical(
                data["exposure_history"], categories=["never", "adopted", "stopped", "persistent"]
            )
            for col in ["education_group", "rural_category", "cohort", "panel_id", "wave_order"]:
                data[col] = data[col].astype(str)
            formula = (
                'next_healthy_aging_binary ~ C(exposure_history, Treatment(reference="never")) + '
                "healthy_aging_binary + " + " + ".join(covariate_terms())
            )
            rows.extend(
                fit_terms(
                    data,
                    formula,
                    terms,
                    {
                        "analysis": "exposure_history",
                        "cohort": cohort,
                        "outcome": "next_healthy_aging_binary",
                        "exposure": exposure,
                    },
                )
            )
    model_rows = pd.DataFrame(rows)
    meta = random_effects_meta(model_rows, ["analysis", "outcome", "exposure", "term"])
    return model_rows, meta


def run_transition_models(lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    transition_specs = [
        ("maintain_healthy", 1.0, "next_healthy_aging_binary"),
        ("gain_healthy", 0.0, "next_healthy_aging_binary"),
    ]
    for transition, baseline_value, outcome in transition_specs:
        for cohort in JOINT_COHORTS:
            data = model_frame(
                lagged,
                [cohort],
                ["internet_use", "social_participation"],
                outcome,
                baseline=None,
                extra_cols=["healthy_aging_binary"],
            )
            data = data[data["healthy_aging_binary"].eq(baseline_value)].copy()
            data = add_joint_exposure(data)
            formula = joint_formula(outcome, baseline=None, pooled=False)
            rows.extend(
                fit_terms(
                    data,
                    formula,
                    JOINT_TERMS,
                    {
                        "analysis": "transition",
                        "transition": transition,
                        "cohort": cohort,
                        "outcome": outcome,
                    },
                )
            )
    model_rows = pd.DataFrame(rows)
    meta = random_effects_meta(model_rows, ["analysis", "transition", "outcome", "term"])
    return model_rows, meta


def run_domain_models(lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for exposure, cohorts in [("internet_use", INTERNET_COHORTS), ("social_participation", SOCIAL_COHORTS)]:
        for domain_label, domain_col in DOMAIN_OUTCOMES.items():
            outcome = f"next_{domain_col}"
            for cohort in cohorts:
                data = model_frame(lagged, [cohort], [exposure], outcome, baseline=domain_col)
                formula = single_exposure_formula(outcome, exposure, baseline=domain_col)
                rows.extend(
                    fit_terms(
                        data,
                        formula,
                        [(exposure, exposure)],
                        {
                            "analysis": "domain_outcome",
                            "domain": domain_label,
                            "cohort": cohort,
                            "outcome": outcome,
                            "exposure": exposure,
                        },
                    )
                )
    model_rows = pd.DataFrame(rows)
    meta = random_effects_meta(model_rows, ["analysis", "domain", "outcome", "exposure", "term"])
    return model_rows, meta


def run_alternative_definitions(lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    outcomes = [
        ("strict_all_available_components", "next_healthy_strict_binary", "healthy_strict_binary"),
        ("lenient_60pct_threshold", "next_healthy_lenient_binary", "healthy_lenient_binary"),
        ("without_cognition_component", "next_healthy_no_cognition_binary", "healthy_no_cognition_binary"),
    ]
    for definition, outcome, baseline in outcomes:
        for exposure, cohorts in [("internet_use", INTERNET_COHORTS), ("social_participation", SOCIAL_COHORTS)]:
            for cohort in cohorts:
                data = model_frame(lagged, [cohort], [exposure], outcome, baseline=baseline)
                formula = single_exposure_formula(outcome, exposure, baseline=baseline)
                rows.extend(
                    fit_terms(
                        data,
                        formula,
                        [(exposure, exposure)],
                        {
                            "analysis": "alternative_definition",
                            "definition": definition,
                            "cohort": cohort,
                            "outcome": outcome,
                            "exposure": exposure,
                        },
                    )
                )
    model_rows = pd.DataFrame(rows)
    meta = random_effects_meta(model_rows, ["analysis", "definition", "outcome", "exposure", "term"])
    return model_rows, meta


def run_subgroup_models(lagged: pd.DataFrame) -> pd.DataFrame:
    data = lagged.copy()
    data["age_group"] = pd.cut(
        data["age"],
        bins=[-np.inf, 59, 69, 79, np.inf],
        labels=["under60", "60-69", "70-79", "80plus"],
    ).astype("object")
    data["sex"] = np.where(data["female"].eq(1), "female", "male")
    data["baseline_health"] = np.where(data["healthy_aging_binary"].eq(1), "healthy", "unhealthy")

    subgroup_specs = [
        ("age_group", "age_group", []),
        ("sex", "sex", ["female"]),
        ("education", "education_group", ["education_group"]),
        ("rurality", "rural_category", ["rural_category"]),
        ("baseline_health", "baseline_health", []),
    ]

    rows = []
    for exposure, cohorts in [("internet_use", INTERNET_COHORTS), ("social_participation", SOCIAL_COHORTS)]:
        for subgroup, col, skip in subgroup_specs:
            levels = [x for x in pd.Series(data[col].dropna().unique()).astype(str).tolist() if x != "unknown"]
            for level in sorted(levels):
                if subgroup == "baseline_health":
                    baseline = None
                    extra_cols = ["healthy_aging_binary", col] if col != "healthy_aging_binary" else ["healthy_aging_binary"]
                else:
                    baseline = "healthy_aging_binary"
                    extra_cols = [col]
                frame = model_frame(
                    data,
                    cohorts,
                    [exposure],
                    "next_healthy_aging_binary",
                    baseline=baseline,
                    extra_cols=extra_cols,
                )
                frame = frame[frame[col].astype(str).eq(level)].copy()
                formula = single_exposure_formula(
                    "next_healthy_aging_binary",
                    exposure,
                    baseline=baseline,
                    pooled=True,
                    skip_covariates=set(skip),
                )
                rows.extend(
                    fit_terms(
                        frame,
                        formula,
                        [(exposure, exposure)],
                        {
                            "analysis": "subgroup_pooled",
                            "subgroup": subgroup,
                            "level": level,
                            "cohort": "pooled",
                            "outcome": "next_healthy_aging_binary",
                            "exposure": exposure,
                        },
                    )
                )
    return pd.DataFrame(rows)


def build_attrition_weights(df: pd.DataFrame, lagged: pd.DataFrame, exposure: str, cohorts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = df[df["cohort"].isin(cohorts)].copy()
    group = source.groupby("panel_id", sort=False)
    source["next_same_cohort"] = group["cohort"].shift(-1)
    source["next_year_observed"] = group["year"].shift(-1)
    source["has_next_observation"] = (
        source["next_same_cohort"].eq(source["cohort"])
        & source["next_year_observed"].notna()
        & source["next_year_observed"].gt(source["year"])
    ).astype(float)
    max_year = source.groupby("cohort")["year"].transform("max")
    source = source[source["year"].lt(max_year)].copy()
    required = [
        "has_next_observation",
        exposure,
        "healthy_aging_binary",
        "age",
        "age_sq",
        "female",
        "married",
        "education_group",
        "rural_category",
        "wave_order",
    ]
    source = source.dropna(subset=required)
    source = source[valid_binary(source[exposure]) & valid_binary(source["healthy_aging_binary"])]
    for col in ["education_group", "rural_category", "cohort", "wave_order", "panel_id"]:
        source[col] = source[col].astype(str)

    diagnostics = {
        "exposure": exposure,
        "cohorts": ",".join(cohorts),
        "attrition_model_n": int(len(source)),
        "observed_next_pct": float(source["has_next_observation"].mean() * 100) if len(source) else np.nan,
        "status": "not_run",
    }
    if len(source) < 500 or source["has_next_observation"].nunique(dropna=True) < 2:
        diagnostics["status"] = "skipped"
        lagged[f"ipw_{exposure}"] = np.nan
        return lagged, pd.DataFrame([diagnostics])

    formula = (
        f"has_next_observation ~ {exposure} + healthy_aging_binary + age + age_sq + female + "
        "C(education_group) + married + C(rural_category) + C(cohort) + C(wave_order)"
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.glm(formula=formula, data=source, family=sm.families.Binomial()).fit()
        source["p_observed"] = np.clip(fitted.predict(source), 0.05, 0.95)
        source[f"ipw_{exposure}"] = 1 / source["p_observed"]
        source[f"ipw_{exposure}"] = source[f"ipw_{exposure}"] / source[f"ipw_{exposure}"].mean()
        weight_map = source.set_index("row_id")[f"ipw_{exposure}"]
        lagged[f"ipw_{exposure}"] = lagged["row_id"].map(weight_map)
        diagnostics.update(
            {
                "status": "ok",
                "mean_weight": float(source[f"ipw_{exposure}"].mean()),
                "p1_weight": float(source[f"ipw_{exposure}"].quantile(0.01)),
                "p99_weight": float(source[f"ipw_{exposure}"].quantile(0.99)),
            }
        )
    except Exception as exc:
        lagged[f"ipw_{exposure}"] = np.nan
        diagnostics["status"] = f"failed: {type(exc).__name__}: {exc}"
    return lagged, pd.DataFrame([diagnostics])


def run_ipw_models(df: pd.DataFrame, lagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    working = lagged.copy()
    for exposure, cohorts in [("internet_use", INTERNET_COHORTS), ("social_participation", SOCIAL_COHORTS)]:
        working, diag = build_attrition_weights(df, working, exposure, cohorts)
        diagnostics.append(diag)
        weight_col = f"ipw_{exposure}"
        for cohort in cohorts:
            data = model_frame(working, [cohort], [exposure], "next_healthy_aging_binary", baseline="healthy_aging_binary")
            data[weight_col] = working.loc[data.index, weight_col]
            data = data.dropna(subset=[weight_col])
            formula = single_exposure_formula("next_healthy_aging_binary", exposure, baseline="healthy_aging_binary")
            rows.extend(
                fit_terms(
                    data,
                    formula,
                    [(exposure, exposure)],
                    {
                        "analysis": "ipw_attrition",
                        "cohort": cohort,
                        "outcome": "next_healthy_aging_binary",
                        "exposure": exposure,
                    },
                    weight_col=weight_col,
                )
            )
    return pd.DataFrame(rows), pd.concat(diagnostics, ignore_index=True)


def leave_one_cohort_out(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    if rows.empty:
        return pd.DataFrame()
    ok = rows[rows["status"].eq("ok")].copy()
    for keys, group in ok.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(by, keys))
        cohorts = sorted(group["cohort"].dropna().unique().tolist())
        for omitted in cohorts:
            subset = group[group["cohort"].ne(omitted)].copy()
            meta = random_effects_meta(subset, by)
            if meta.empty:
                out.append({**base, "omitted_cohort": omitted, "status": "skipped"})
            else:
                row = meta.iloc[0].to_dict()
                row["omitted_cohort"] = omitted
                out.append(row)
    return pd.DataFrame(out)


def plot_joint_forest(model_rows: pd.DataFrame, meta: pd.DataFrame, output_path: Path) -> None:
    ok = model_rows[model_rows["status"].eq("ok")].copy()
    if ok.empty:
        return
    terms = ["internet_only", "social_only", "both"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, term in zip(axes, terms):
        rows = ok[ok["term"].eq(term)].sort_values("odds_ratio")
        y = np.arange(len(rows))
        ax.errorbar(
            rows["odds_ratio"],
            y,
            xerr=[rows["odds_ratio"] - rows["ci_low"], rows["ci_high"] - rows["odds_ratio"]],
            fmt="o",
            color="#245c6f",
            ecolor="#9bb9c4",
            capsize=3,
        )
        meta_row = meta[meta["term"].eq(term)]
        if not meta_row.empty and meta_row.iloc[0].get("status") == "ok":
            pooled = float(meta_row.iloc[0]["odds_ratio"])
            ax.axvline(pooled, color="#b23b3b", linewidth=1.4)
        ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(term.replace("_", " "))
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(rows["cohort"])
        else:
            ax.set_yticklabels([])
    axes[1].set_xlabel("Odds ratio for next-wave healthy ageing")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_subgroup_heatmap(subgroups: pd.DataFrame, output_path: Path) -> None:
    ok = subgroups[subgroups["status"].eq("ok")].copy()
    if ok.empty:
        return
    ok["row_label"] = ok["subgroup"].astype(str) + ": " + ok["level"].astype(str)
    ok["col_label"] = ok["exposure"].astype(str)
    pivot = ok.pivot_table(index="row_label", columns="col_label", values="odds_ratio", aggfunc="first")
    pivot = pivot.sort_index()
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, max(4, len(pivot) * 0.32)))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=0.7, vmax=1.7)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Pooled subgroup odds ratios")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("OR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_robustness(meta_main: pd.DataFrame, alt_meta: pd.DataFrame, ipw_meta: pd.DataFrame, output_path: Path) -> None:
    rows = []
    if not meta_main.empty:
        for _, row in meta_main.iterrows():
            if row.get("status") == "ok":
                rows.append(
                    {
                        "analysis": "main_binary",
                        "exposure": row.get("exposure"),
                        "or": row.get("odds_ratio"),
                        "low": row.get("ci_low"),
                        "high": row.get("ci_high"),
                    }
                )
    for _, row in alt_meta.iterrows():
        if row.get("status") == "ok":
            rows.append(
                {
                    "analysis": str(row.get("definition")),
                    "exposure": row.get("exposure"),
                    "or": row.get("odds_ratio"),
                    "low": row.get("ci_low"),
                    "high": row.get("ci_high"),
                }
            )
    for _, row in ipw_meta.iterrows():
        if row.get("status") == "ok":
            rows.append(
                {
                    "analysis": "ipw_attrition",
                    "exposure": row.get("exposure"),
                    "or": row.get("odds_ratio"),
                    "low": row.get("ci_low"),
                    "high": row.get("ci_high"),
                }
            )
    plot_df = pd.DataFrame(rows).dropna(subset=["or"])
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["exposure"].astype(str) + " | " + plot_df["analysis"].astype(str)
    plot_df = plot_df.sort_values(["exposure", "analysis"])
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.34)))
    ax.errorbar(
        plot_df["or"],
        y,
        xerr=[plot_df["or"] - plot_df["low"], plot_df["high"] - plot_df["or"]],
        fmt="o",
        color="#394d7f",
        ecolor="#aab4d4",
        capsize=3,
    )
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("Odds ratio")
    ax.set_title("Robustness across healthy ageing definitions and IPW")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_report(
    output_dir: Path,
    joint_meta: pd.DataFrame,
    interaction_meta: pd.DataFrame,
    reri_meta: pd.DataFrame,
    history_meta: pd.DataFrame,
    transition_meta: pd.DataFrame,
    domain_meta: pd.DataFrame,
    alt_meta: pd.DataFrame,
    ipw_meta: pd.DataFrame,
    subgroup_rows: pd.DataFrame,
    evalues: pd.DataFrame,
) -> None:
    def fmt(x: object, digits: int = 2) -> str:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"

    lines = [
        "# NC 升级分析结果包",
        "",
        "目标：把第一版主效应升级为 connectedness resources 与 healthy ageing trajectories 的多层证据链。",
        "",
        "## 1. 联合暴露作为中心发现",
        "",
        "| Contrast vs neither | K | OR | 95% CI | I2 % |",
        "|---|---:|---:|---|---:|",
    ]
    for _, row in joint_meta.iterrows():
        lines.append(
            f"| {row.get('term')} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )

    lines += [
        "",
        "## 2. 联合暴露交互",
        "",
        "| Metric | K | Estimate | 95% CI | I2 % |",
        "|---|---:|---:|---|---:|",
    ]
    for _, row in interaction_meta.iterrows():
        lines.append(
            f"| multiplicative interaction OR | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )
    for _, row in reri_meta.iterrows():
        lines.append(
            f"| additive interaction RERI | {int(row.get('k', 0))} | {fmt(row.get('reri'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )

    lines += [
        "",
        "## 3. 数字/社会参与动态暴露",
        "",
        "| Exposure | History vs never | K | OR | 95% CI | I2 % |",
        "|---|---|---:|---:|---|---:|",
    ]
    for _, row in history_meta.iterrows():
        lines.append(
            f"| {row.get('exposure')} | {row.get('term')} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )

    lines += [
        "",
        "## 4. 健康老龄化状态转移",
        "",
        "| Transition | Contrast | K | OR | 95% CI | I2 % |",
        "|---|---|---:|---:|---|---:|",
    ]
    for _, row in transition_meta.iterrows():
        lines.append(
            f"| {row.get('transition')} | {row.get('term')} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )

    lines += [
        "",
        "## 5. 健康域驱动结果",
        "",
        "下面列出每个健康域的随机效应 meta。若某健康域结果强，后续可作为机制/解释重点。",
        "",
        "| Exposure | Domain | K | OR | 95% CI | I2 % |",
        "|---|---|---:|---:|---|---:|",
    ]
    for _, row in domain_meta.iterrows():
        lines.append(
            f"| {row.get('exposure')} | {row.get('domain')} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )

    lines += [
        "",
        "## 6. 稳健性分析",
        "",
        "| Analysis | Exposure | K | OR | 95% CI | I2 % |",
        "|---|---|---:|---:|---|---:|",
    ]
    for _, row in alt_meta.iterrows():
        lines.append(
            f"| {row.get('definition')} | {row.get('exposure')} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )
    for _, row in ipw_meta.iterrows():
        lines.append(
            f"| ipw_attrition | {row.get('exposure')} | {int(row.get('k', 0))} | {fmt(row.get('odds_ratio'))} | "
            f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))} | {fmt(row.get('i2_pct'))} |"
        )

    lines += [
        "",
        "## 7. 异质性信号",
        "",
        f"- 已运行 pooled subgroup models：{len(subgroup_rows)} 条模型记录。",
        "- 重点看 `subgroup_pooled_models.csv` 和 `figure_subgroup_heatmap.png`。",
        "- 如果某些分层 OR 明显不同，下一轮应改成分队列 subgroup meta，而不是只用 pooled 模型。",
        "",
        "## 8. E-value",
        "",
        "| Analysis | Term/Exposure | OR | CI low | E-value point | E-value CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in evalues.iterrows():
        label = row.get("term")
        if pd.isna(label):
            label = row.get("exposure")
        if pd.isna(label):
            label = row.get("definition")
        if pd.isna(label):
            label = ""
        analysis = row.get("analysis", row.get("definition", "main"))
        lines.append(
            f"| {analysis} | {label} | {fmt(row.get('odds_ratio'))} | {fmt(row.get('ci_low'))} | "
            f"{fmt(row.get('e_value_point'))} | {fmt(row.get('e_value_ci'))} |"
        )

    lines += [
        "",
        "## 9. 现在距离 NC 还差什么",
        "",
        "- 必须把 joint exposure 提到主结果，而不是补充模型。",
        "- 必须用状态转移结果讲 trajectories：maintain healthy 和 gain healthy。",
        "- 动态暴露结果用于回答 adoption/persistence，比静态 internet use 更有论文新意。",
        "- RERI 若不显著，不能硬写协同；可以写 both 组风险优势最大，但 additive interaction 证据有限。",
        "- 高 I2 不是小问题，需要在正文中解释为跨国家/队列差异，并用 subgroup/leave-one-out 支撑。",
        "- IPW 已作为第一层失访校正；若后续能接死亡/退出状态，应补 competing risk 或死亡复合结局。",
        "- domain outcome 是机制线索，不应直接写成中介因果；正式投稿前可进一步做 t -> mediator -> t+2 的探索性 mediation。",
    ]
    (output_dir / "nc_upgrade_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path, columns=CORE_COLS)
    df = clean_for_upgrade(df)
    lagged = build_lagged_table(df)

    joint_rows, joint_meta = run_joint_by_cohort(lagged)
    interaction_rows, interaction_meta, reri_meta = run_joint_interaction_models(lagged)
    history_rows, history_meta = run_exposure_history_models(lagged)
    transition_rows, transition_meta = run_transition_models(lagged)
    domain_rows, domain_meta = run_domain_models(lagged)
    alt_rows, alt_meta = run_alternative_definitions(lagged)
    subgroup_rows = run_subgroup_models(lagged)
    ipw_rows, ipw_diag = run_ipw_models(df, lagged)
    ipw_meta = random_effects_meta(ipw_rows, ["analysis", "outcome", "exposure", "term"])

    joint_loo = leave_one_cohort_out(joint_rows, ["analysis", "outcome", "term"])
    history_loo = leave_one_cohort_out(history_rows, ["analysis", "outcome", "exposure", "term"])
    domain_loo = leave_one_cohort_out(domain_rows, ["analysis", "domain", "outcome", "exposure", "term"])

    main_meta_path = LONGITUDINAL_OUTPUT_DIR / "random_effects_meta_analysis.csv"
    if main_meta_path.exists():
        main_meta = pd.read_csv(main_meta_path)
        main_meta["analysis"] = "main_binary"
    else:
        main_meta = pd.DataFrame()

    evalues = pd.concat(
        [
            build_e_values(joint_meta, ["analysis", "term"]),
            build_e_values(transition_meta, ["analysis", "transition", "term"]),
            build_e_values(history_meta, ["analysis", "exposure", "term"]),
            build_e_values(alt_meta, ["analysis", "definition", "exposure"]),
            build_e_values(ipw_meta, ["analysis", "exposure"]),
        ],
        ignore_index=True,
    )

    joint_rows.to_csv(output_dir / "model_joint_exposure_by_cohort.csv", index=False, encoding="utf-8-sig")
    joint_meta.to_csv(output_dir / "meta_joint_exposure.csv", index=False, encoding="utf-8-sig")
    interaction_rows.to_csv(output_dir / "model_joint_interaction_reri.csv", index=False, encoding="utf-8-sig")
    interaction_meta.to_csv(output_dir / "meta_joint_multiplicative_interaction.csv", index=False, encoding="utf-8-sig")
    reri_meta.to_csv(output_dir / "meta_joint_additive_interaction_reri.csv", index=False, encoding="utf-8-sig")
    history_rows.to_csv(output_dir / "model_exposure_history.csv", index=False, encoding="utf-8-sig")
    history_meta.to_csv(output_dir / "meta_exposure_history.csv", index=False, encoding="utf-8-sig")
    joint_loo.to_csv(output_dir / "leave_one_cohort_out_joint.csv", index=False, encoding="utf-8-sig")
    history_loo.to_csv(output_dir / "leave_one_cohort_out_exposure_history.csv", index=False, encoding="utf-8-sig")
    transition_rows.to_csv(output_dir / "model_transition_joint_exposure.csv", index=False, encoding="utf-8-sig")
    transition_meta.to_csv(output_dir / "meta_transition_joint_exposure.csv", index=False, encoding="utf-8-sig")
    domain_rows.to_csv(output_dir / "model_domain_outcomes.csv", index=False, encoding="utf-8-sig")
    domain_meta.to_csv(output_dir / "meta_domain_outcomes.csv", index=False, encoding="utf-8-sig")
    domain_loo.to_csv(output_dir / "leave_one_cohort_out_domain.csv", index=False, encoding="utf-8-sig")
    alt_rows.to_csv(output_dir / "model_alternative_definitions.csv", index=False, encoding="utf-8-sig")
    alt_meta.to_csv(output_dir / "meta_alternative_definitions.csv", index=False, encoding="utf-8-sig")
    subgroup_rows.to_csv(output_dir / "subgroup_pooled_models.csv", index=False, encoding="utf-8-sig")
    ipw_rows.to_csv(output_dir / "model_ipw_attrition.csv", index=False, encoding="utf-8-sig")
    ipw_meta.to_csv(output_dir / "meta_ipw_attrition.csv", index=False, encoding="utf-8-sig")
    ipw_diag.to_csv(output_dir / "ipw_attrition_diagnostics.csv", index=False, encoding="utf-8-sig")
    evalues.to_csv(output_dir / "e_values.csv", index=False, encoding="utf-8-sig")

    plot_joint_forest(joint_rows, joint_meta, output_dir / "figure_joint_exposure_forest.png")
    plot_subgroup_heatmap(subgroup_rows, output_dir / "figure_subgroup_heatmap.png")
    plot_robustness(main_meta, alt_meta, ipw_meta, output_dir / "figure_robustness_grid.png")

    write_report(
        output_dir,
        joint_meta,
        interaction_meta,
        reri_meta,
        history_meta,
        transition_meta,
        domain_meta,
        alt_meta,
        ipw_meta,
        subgroup_rows,
        evalues,
    )

    print(f"Wrote NC upgrade outputs to {output_dir}")
    print("\nJoint exposure meta:")
    print(joint_meta.to_string(index=False))
    print("\nJoint interaction meta:")
    print(interaction_meta.to_string(index=False))
    print("\nRERI meta:")
    print(reri_meta.to_string(index=False))
    print("\nExposure history meta:")
    print(history_meta.to_string(index=False))
    print("\nTransition meta:")
    print(transition_meta.to_string(index=False))
    print("\nAlternative definitions meta:")
    print(alt_meta.to_string(index=False))
    print("\nIPW meta:")
    print(ipw_meta.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
