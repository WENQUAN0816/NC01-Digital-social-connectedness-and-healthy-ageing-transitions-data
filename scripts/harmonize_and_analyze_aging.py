"""
First-pass harmonization and exploratory analysis for seven HRS-family ageing cohorts.

The script reads the simplified CSV files on F:, maps comparable variables into a
common person-wave table, builds a pragmatic healthy-ageing score, and exports
descriptive tables plus exploratory pooled/country-specific logistic models.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DEFAULT_DATA_DIR = Path(r"F:\目前养老官方数据库\七国包括charls（非常简单明了的数据）")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "analysis" / "outputs"


@dataclass(frozen=True)
class CohortConfig:
    cohort: str
    filename: str
    id_code: str
    wave_code: str | None = None
    year_code: str | None = None
    age_code: str | None = None
    sex_code: str | None = None
    education_code: str | None = None
    marital_code: str | None = None
    rural_code: str | None = None
    srh_code: str | None = None
    adl_score_code: str | None = None
    iadl_score_code: str | None = None
    adl_item_codes: tuple[str, ...] = ()
    iadl_item_codes: tuple[str, ...] = ()
    depression_score_code: str | None = None
    depression_binary_code: str | None = None
    cognition_code: str | None = None
    frailty_code: str | None = None
    internet_code: str | None = None
    internet_frequency_code: str | None = None
    social_code: str | None = None
    social_frequency_code: str | None = None
    work_code: str | None = None
    retired_code: str | None = None
    disease_codes: tuple[str, ...] = ()


COHORTS = [
    CohortConfig(
        cohort="CHARLS",
        filename="charls.csv",
        id_code="ID",
        wave_code="wave",
        year_code="iwy",
        age_code="age",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="marry",
        rural_code="hrural",
        srh_code="srh",
        adl_score_code="adlab_c",
        iadl_score_code="iadl",
        depression_score_code="cesd10",
        cognition_code="tcog_z_z",
        frailty_code="frailtyb",
        internet_code="social10",
        social_code="socwk",
        work_code="work",
        retired_code="retire",
        disease_codes=("hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "kidneye", "arthre", "psyche"),
    ),
    CohortConfig(
        cohort="ELSA",
        filename="elsa.csv",
        id_code="idauniqc",
        wave_code="wave",
        year_code="iwindy",
        age_code="agey",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="mstath",
        srh_code="shlt",
        adl_score_code="adltot6",
        iadl_score_code="iadltot2_e",
        depression_score_code="cesd",
        depression_binary_code="depressive",
        cognition_code="tcog_z_z",
        frailty_code="frailty",
        internet_code="internet",
        social_code="group6",
        work_code="work",
        disease_codes=("hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "arthre", "psyche"),
    ),
    CohortConfig(
        cohort="HRS",
        filename="hrs 截止到2020.csv",
        id_code="hhidpn",
        wave_code="wave",
        year_code="iwendy",
        age_code="ragey_m",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="mstath",
        rural_code="rural",
        srh_code="shlt",
        adl_score_code="adl6a",
        iadl_score_code="iadl5a",
        depression_score_code="cesd",
        depression_binary_code="depressive",
        cognition_code="tcog_z_z",
        frailty_code="frailty",
        internet_code="email",
        social_code="socwk",
        work_code="work",
        retired_code="sayret",
        disease_codes=("hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "arthre", "psyche"),
    ),
    CohortConfig(
        cohort="KLoSA",
        filename="klosa 截止到2020 .csv",
        id_code="pid",
        wave_code="wave",
        year_code="iwy",
        age_code="agey",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="mstath",
        rural_code="rural",
        srh_code="shlt",
        adl_item_codes=("dressb", "bathb", "eatb", "toiltb", "bedb_k", "urinb"),
        iadl_item_codes=("mealsb", "shopb", "medsb", "moneyb", "phoneb"),
        depression_score_code="cesd10b",
        cognition_code="cog_total",
        social_code="act2",
        social_frequency_code="freq_2",
        work_code="work",
        disease_codes=("hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "arthre", "psyche"),
    ),
    CohortConfig(
        cohort="LASI",
        filename="lasi.csv",
        id_code="prim_key",
        year_code="r1iwy",
        age_code="r1agey",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="r1mstath",
        rural_code="hh1rural",
        srh_code="r1shlt",
        adl_score_code="r1adltot6",
        iadl_score_code="r1iadltot_l",
        depression_score_code="r1cesd10",
        depression_binary_code="r1cesd10dep",
        cognition_code="r1cog_total",
        internet_frequency_code="r1act11",
        social_code="r1socwk",
        work_code="r1work",
        retired_code="r1sayret_l",
        disease_codes=("r1hibpe", "r1diabe", "r1cancre", "r1lunge", "r1hearte", "r1stroke", "r1arthre", "r1psyche"),
    ),
    CohortConfig(
        cohort="MHAS",
        filename="mhas.csv",
        id_code="rahhidnp",
        wave_code="wave",
        year_code="iwy",
        age_code="agey",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="mstath",
        rural_code="rural",
        srh_code="shlt",
        adl_score_code="adltot6",
        iadl_score_code="iadlfour",
        depression_score_code="cesd_m",
        work_code="work",
        disease_codes=("hibpe", "diabe", "cancre", "respe", "hearte", "stroke", "arthre"),
        cognition_code="tr16",
    ),
    CohortConfig(
        cohort="SHARE",
        filename="share.csv",
        id_code="mergeid",
        wave_code="wave",
        year_code="iwy",
        age_code="agey",
        sex_code="ragender",
        education_code="raeducl",
        marital_code="mstath",
        rural_code="rural",
        srh_code="shlt",
        adl_item_codes=("walkra", "dressa", "batha", "eata", "beda", "toilta"),
        iadl_item_codes=("phonea", "medsa", "moneya", "shopa", "mealsa"),
        frailty_code="frailtyb",
        internet_code="internet",
        social_code="act3",
        social_frequency_code="freq_act3",
        work_code="work",
        cognition_code="tr20",
        disease_codes=("hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "kidneye", "arthre", "psyche"),
    ),
]


MISSING_VALUES = {
    "",
    "NA",
    "NaN",
    "nan",
    "None",
    ".",
    ".m:Missing",
    ".d:DK",
    ".r:Refuse",
    ".p:Proxy",
    "不相关和缺失",
    "未知",
}


def code_of(column_name: str) -> str:
    return column_name.split(" ", 1)[0].strip()


def clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.mask(out.isin(MISSING_VALUES))
    return out


def to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = clean_string(series)
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def yes_no(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = clean_string(series)
    low = s.str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    positive = (
        s.eq("是")
        | low.eq("yes")
        | low.eq("true")
        | low.eq("1")
        | low.str.startswith("1.", na=False)
        | low.str.contains("yes", na=False)
    )
    negative = (
        s.eq("否")
        | low.eq("no")
        | low.eq("false")
        | low.eq("0")
        | low.str.startswith("0.", na=False)
        | low.str.contains("no", na=False)
    )
    out.loc[positive] = 1.0
    out.loc[negative] = 0.0
    return out


def frequency_positive(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = clean_string(series)
    low = s.str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    negative = (
        s.eq("否")
        | s.eq("从不")
        | low.eq("0")
        | low.str.startswith("0.", na=False)
        | low.str.contains("no", na=False)
    )
    positive = s.notna() & ~negative
    out.loc[negative] = 0.0
    out.loc[positive] = 1.0
    return out


def education_group(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    s = clean_string(series)
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out.loc[s.str.contains("高等|higher|college|university", case=False, na=False)] = "high"
    out.loc[s.str.contains("高中|职业|high", case=False, na=False)] = "middle"
    out.loc[s.str.contains("低于|以下|below|less", case=False, na=False)] = "low"
    return out


def married_binary(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = clean_string(series)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    yes = s.str.contains("已婚|伴侣|同居|注册伴侣", na=False)
    no = s.str.contains("丧偶|寡妇|离婚|分居|从未|被遗弃", na=False)
    out.loc[yes] = 1.0
    out.loc[no] = 0.0
    return out


def rural_binary(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = clean_string(series)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[s.str.contains("农村", na=False)] = 1.0
    out.loc[s.str.contains("城市|城镇|都市", na=False)] = 0.0
    return out


def fair_or_poor_srh(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = clean_string(series)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    fair_or_poor = s.str.contains("一般|差|较差|很差", na=False)
    better = s.str.contains("好|较好|很好|非常好", na=False) & ~fair_or_poor
    out.loc[fair_or_poor] = 1.0
    out.loc[better] = 0.0
    return out


def retired_binary(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = clean_string(series)
    out = yes_no(s)
    retired = s.str.contains("退休", na=False) & ~s.str.contains("没有退休|不相关|缺失", na=False)
    not_retired = s.str.contains("没有退休", na=False)
    out.loc[retired] = 1.0
    out.loc[not_retired] = 0.0
    return out


def sum_binary_items(data: pd.DataFrame, codes: Iterable[str], code_map: dict[str, str]) -> pd.Series:
    items = []
    for code in codes:
        if code in code_map:
            items.append(yes_no(data[code_map[code]]))
    if not items:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    mat = pd.concat(items, axis=1)
    score = mat.sum(axis=1, min_count=1)
    return score.astype("float64")


def coalesce_numeric(data: pd.DataFrame, code_map: dict[str, str], codes: Iterable[str]) -> pd.Series:
    out = pd.Series(np.nan, index=data.index, dtype="float64")
    for code in codes:
        if code in code_map:
            values = to_numeric(data[code_map[code]])
            out = out.combine_first(values)
    return out


def get_series(data: pd.DataFrame, code_map: dict[str, str], code: str | None) -> pd.Series | None:
    if code and code in code_map:
        return data[code_map[code]]
    return None


def transform_chunk(chunk: pd.DataFrame, config: CohortConfig, code_map: dict[str, str]) -> pd.DataFrame:
    idx = chunk.index

    def g(code: str | None) -> pd.Series | None:
        return get_series(chunk, code_map, code)

    out = pd.DataFrame(index=idx)
    out["cohort"] = config.cohort
    out["participant_id"] = clean_string(g(config.id_code)) if g(config.id_code) is not None else pd.NA
    out["panel_id"] = config.cohort + ":" + out["participant_id"].astype("string")
    out["wave"] = clean_string(g(config.wave_code)) if g(config.wave_code) is not None else pd.NA
    if config.wave_code is None:
        out["wave"] = "wave1"
    out["year"] = to_numeric(g(config.year_code)) if g(config.year_code) is not None else np.nan
    out["age"] = to_numeric(g(config.age_code)) if g(config.age_code) is not None else np.nan
    sex = clean_string(g(config.sex_code)) if g(config.sex_code) is not None else pd.Series(pd.NA, index=idx)
    out["female"] = np.where(sex.str.contains("女性", na=False), 1.0, np.where(sex.str.contains("男性", na=False), 0.0, np.nan))
    out["education_group"] = education_group(g(config.education_code))
    out["married"] = married_binary(g(config.marital_code))
    out["rural"] = rural_binary(g(config.rural_code))
    out["rural_category"] = "unknown"
    out.loc[out["rural"].eq(1.0), "rural_category"] = "rural"
    out.loc[out["rural"].eq(0.0), "rural_category"] = "urban"
    out["fair_or_poor_srh"] = fair_or_poor_srh(g(config.srh_code))

    if config.adl_score_code and config.adl_score_code in code_map:
        out["adl_score"] = to_numeric(chunk[code_map[config.adl_score_code]])
    else:
        out["adl_score"] = sum_binary_items(chunk, config.adl_item_codes, code_map)
    if config.iadl_score_code and config.iadl_score_code in code_map:
        out["iadl_score"] = to_numeric(chunk[code_map[config.iadl_score_code]])
    else:
        out["iadl_score"] = sum_binary_items(chunk, config.iadl_item_codes, code_map)
    out["adl_limitation"] = np.where(out["adl_score"].notna(), (out["adl_score"] > 0).astype(float), np.nan)
    out["iadl_limitation"] = np.where(out["iadl_score"].notna(), (out["iadl_score"] > 0).astype(float), np.nan)

    disease_cols = []
    for disease_code in config.disease_codes:
        if disease_code in code_map:
            disease_cols.append(yes_no(chunk[code_map[disease_code]]))
    if disease_cols:
        diseases = pd.concat(disease_cols, axis=1)
        out["chronic_count"] = diseases.sum(axis=1, min_count=1)
    else:
        out["chronic_count"] = np.nan
    out["multimorbidity"] = np.where(out["chronic_count"].notna(), (out["chronic_count"] >= 2).astype(float), np.nan)

    out["depression_score"] = to_numeric(g(config.depression_score_code)) if g(config.depression_score_code) is not None else np.nan
    if g(config.depression_binary_code) is not None:
        out["depressive_symptoms"] = yes_no(g(config.depression_binary_code))
    else:
        threshold = 10.0
        if config.cohort in {"ELSA", "HRS"}:
            threshold = 4.0
        if config.cohort == "MHAS":
            threshold = 5.0
        out["depressive_symptoms"] = np.where(
            pd.notna(out["depression_score"]),
            (out["depression_score"] >= threshold).astype(float),
            np.nan,
        )

    out["cognition_value"] = to_numeric(g(config.cognition_code)) if g(config.cognition_code) is not None else np.nan
    out["frailty_index"] = to_numeric(g(config.frailty_code)) if g(config.frailty_code) is not None else np.nan
    out["frail_25pct"] = np.where(out["frailty_index"].notna(), (out["frailty_index"] >= 25).astype(float), np.nan)

    if g(config.internet_code) is not None:
        out["internet_use"] = yes_no(g(config.internet_code))
    elif g(config.internet_frequency_code) is not None:
        out["internet_use"] = frequency_positive(g(config.internet_frequency_code))
    else:
        out["internet_use"] = np.nan

    if g(config.social_code) is not None:
        out["social_participation"] = yes_no(g(config.social_code))
    elif g(config.social_frequency_code) is not None:
        out["social_participation"] = frequency_positive(g(config.social_frequency_code))
    else:
        out["social_participation"] = np.nan
    if out["social_participation"].isna().all() and g(config.social_frequency_code) is not None:
        out["social_participation"] = frequency_positive(g(config.social_frequency_code))

    out["work"] = yes_no(g(config.work_code))
    out["retired"] = retired_binary(g(config.retired_code))
    return out


def read_and_harmonize(data_dir: Path, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    mapping_rows = []
    for config in COHORTS:
        path = data_dir / config.filename
        if not path.exists():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        code_map = {code_of(col): col for col in header}
        wanted_codes = {
            config.id_code,
            config.wave_code,
            config.year_code,
            config.age_code,
            config.sex_code,
            config.education_code,
            config.marital_code,
            config.rural_code,
            config.srh_code,
            config.adl_score_code,
            config.iadl_score_code,
            config.depression_score_code,
            config.depression_binary_code,
            config.cognition_code,
            config.frailty_code,
            config.internet_code,
            config.internet_frequency_code,
            config.social_code,
            config.social_frequency_code,
            config.work_code,
            config.retired_code,
        }
        wanted_codes.update(config.adl_item_codes)
        wanted_codes.update(config.iadl_item_codes)
        wanted_codes.update(config.disease_codes)
        usecols = [code_map[c] for c in sorted(c for c in wanted_codes if c and c in code_map)]

        for target, source_code in [
            ("id", config.id_code),
            ("wave", config.wave_code),
            ("year", config.year_code),
            ("age", config.age_code),
            ("sex", config.sex_code),
            ("education", config.education_code),
            ("marital", config.marital_code),
            ("rural", config.rural_code),
            ("self_rated_health", config.srh_code),
            ("adl_score", config.adl_score_code),
            ("iadl_score", config.iadl_score_code),
            ("depression_score", config.depression_score_code),
            ("depression_binary", config.depression_binary_code),
            ("cognition", config.cognition_code),
            ("frailty", config.frailty_code),
            ("internet", config.internet_code or config.internet_frequency_code),
            ("social", config.social_code or config.social_frequency_code),
            ("work", config.work_code),
            ("retired", config.retired_code),
        ]:
            mapping_rows.append(
                {
                    "cohort": config.cohort,
                    "target_variable": target,
                    "source_code": source_code or "",
                    "source_column": code_map.get(source_code, "") if source_code else "",
                }
            )

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype="string", encoding="utf-8-sig"):
            frames.append(transform_chunk(chunk, config, code_map))

    harmonized = pd.concat(frames, ignore_index=True)
    mapping = pd.DataFrame(mapping_rows)
    return harmonized, mapping


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Harmonize cognition into a within-cohort z score because raw cognitive scales differ by study.
    df["cognition_z"] = df.groupby("cohort")["cognition_value"].transform(
        lambda s: (s - s.mean(skipna=True)) / s.std(skipna=True) if s.notna().sum() > 2 and s.std(skipna=True) else np.nan
    )
    components = pd.DataFrame(index=df.index)
    components["good_srh"] = np.where(df["fair_or_poor_srh"].notna(), 1 - df["fair_or_poor_srh"], np.nan)
    components["no_adl_limitation"] = np.where(df["adl_limitation"].notna(), 1 - df["adl_limitation"], np.nan)
    components["no_multimorbidity"] = np.where(df["multimorbidity"].notna(), 1 - df["multimorbidity"], np.nan)
    components["not_depressive"] = np.where(df["depressive_symptoms"].notna(), 1 - df["depressive_symptoms"], np.nan)
    components["cognition_not_low"] = np.where(df["cognition_z"].notna(), (df["cognition_z"] > -1).astype(float), np.nan)
    df["healthy_aging_components"] = components.notna().sum(axis=1)
    df["healthy_aging_score"] = components.sum(axis=1, min_count=1)
    df["healthy_aging_ratio"] = df["healthy_aging_score"] / df["healthy_aging_components"]
    df["healthy_aging_binary"] = np.where(
        df["healthy_aging_components"] >= 4,
        (df["healthy_aging_ratio"] >= 0.8).astype(float),
        np.nan,
    )
    return df


def pct(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return math.nan
    return float(valid.mean() * 100)


def build_descriptives(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for cohort, group in df.groupby("cohort", sort=False):
        rows.append(
            {
                "cohort": cohort,
                "person_waves": len(group),
                "unique_people": group["panel_id"].nunique(dropna=True),
                "mean_age": group["age"].mean(skipna=True),
                "female_pct": pct(group["female"]),
                "rural_pct": pct(group["rural"]),
                "married_pct": pct(group["married"]),
                "internet_use_pct": pct(group["internet_use"]),
                "social_participation_pct": pct(group["social_participation"]),
                "work_pct": pct(group["work"]),
                "fair_or_poor_srh_pct": pct(group["fair_or_poor_srh"]),
                "adl_limitation_pct": pct(group["adl_limitation"]),
                "iadl_limitation_pct": pct(group["iadl_limitation"]),
                "multimorbidity_pct": pct(group["multimorbidity"]),
                "depressive_symptoms_pct": pct(group["depressive_symptoms"]),
                "healthy_aging_pct": pct(group["healthy_aging_binary"]),
                "mean_frailty_index": group["frailty_index"].mean(skipna=True),
            }
        )
    overview = pd.DataFrame(rows)

    wave_counts = (
        df.groupby(["cohort", "wave"], dropna=False)
        .agg(person_waves=("panel_id", "size"), unique_people=("panel_id", "nunique"), mean_age=("age", "mean"))
        .reset_index()
    )

    availability_rows = []
    key_cols = [
        "age",
        "female",
        "education_group",
        "married",
        "rural",
        "rural_category",
        "fair_or_poor_srh",
        "adl_score",
        "iadl_score",
        "chronic_count",
        "depressive_symptoms",
        "cognition_z",
        "frailty_index",
        "internet_use",
        "social_participation",
        "work",
        "healthy_aging_binary",
    ]
    for cohort, group in df.groupby("cohort", sort=False):
        for col in key_cols:
            availability_rows.append(
                {
                    "cohort": cohort,
                    "variable": col,
                    "valid_n": int(group[col].notna().sum()),
                    "valid_pct": float(group[col].notna().mean() * 100),
                }
            )
    availability = pd.DataFrame(availability_rows)
    return overview, wave_counts, availability


def fit_models(df: pd.DataFrame, exposure: str) -> pd.DataFrame:
    needed = [
        "healthy_aging_binary",
        exposure,
        "age",
        "female",
        "education_group",
        "married",
        "rural_category",
        "work",
        "cohort",
        "panel_id",
    ]
    model_data = df[needed].copy()
    model_data["education_group"] = model_data["education_group"].astype("object").fillna("unknown")
    model_data["rural_category"] = model_data["rural_category"].astype("object").fillna("unknown")
    model_data = model_data.dropna(
        subset=["healthy_aging_binary", exposure, "age", "female", "married", "work", "cohort", "panel_id"]
    )
    model_data = model_data[(model_data["healthy_aging_binary"].isin([0.0, 1.0])) & (model_data[exposure].isin([0.0, 1.0]))]
    for col in ["education_group", "rural_category", "cohort", "panel_id"]:
        model_data[col] = model_data[col].astype(str)
    results = []

    def fit_one(data: pd.DataFrame, label: str, formula: str) -> None:
        if len(data) < 500 or data[exposure].nunique() < 2 or data["healthy_aging_binary"].nunique() < 2:
            results.append({"model": label, "exposure": exposure, "n": len(data), "status": "skipped_insufficient_variation"})
            return
        try:
            fit = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": data["panel_id"]}
            )
        except Exception:
            fit = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(cov_type="HC1")
        coef = fit.params.get(exposure, np.nan)
        se = fit.bse.get(exposure, np.nan)
        results.append(
            {
                "model": label,
                "exposure": exposure,
                "outcome": "healthy_aging_binary",
                "n": int(fit.nobs),
                "coef_log_odds": coef,
                "se": se,
                "odds_ratio": float(np.exp(coef)) if pd.notna(coef) else np.nan,
                "ci_low": float(np.exp(coef - 1.96 * se)) if pd.notna(coef) and pd.notna(se) else np.nan,
                "ci_high": float(np.exp(coef + 1.96 * se)) if pd.notna(coef) and pd.notna(se) else np.nan,
                "p_value": fit.pvalues.get(exposure, np.nan),
                "status": "ok",
            }
        )

    pooled_formula = (
        f"healthy_aging_binary ~ {exposure} + age + female + C(education_group) + married + C(rural_category) + work + C(cohort)"
    )
    fit_one(model_data, "pooled_with_cohort_fixed_effects", pooled_formula)

    cohort_formula = f"healthy_aging_binary ~ {exposure} + age + female + C(education_group) + married + C(rural_category) + work"
    for cohort, group in model_data.groupby("cohort"):
        fit_one(group, f"{cohort}_specific", cohort_formula)
    return pd.DataFrame(results)


def write_report(
    output_dir: Path,
    overview: pd.DataFrame,
    availability: pd.DataFrame,
    internet_models: pd.DataFrame,
    social_models: pd.DataFrame,
) -> None:
    def fmt(x: float, digits: int = 2) -> str:
        if pd.isna(x):
            return "NA"
        return f"{x:.{digits}f}"

    lines = [
        "# 七国 HRS-family 数据第一版探索分析",
        "",
        "本报告由 `scripts/harmonize_and_analyze_aging.py` 自动生成，目标是先验证 CHARLS/HRS/ELSA/SHARE/KLoSA/LASI/MHAS 简化数据能否按 Nature Communications 常见的多队列套路跑通。",
        "",
        "## 样本概览",
        "",
        "| Cohort | Person-waves | People | Mean age | Female % | Internet % | Social % | Healthy ageing % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overview.iterrows():
        lines.append(
            f"| {row['cohort']} | {int(row['person_waves'])} | {int(row['unique_people'])} | "
            f"{fmt(row['mean_age'])} | {fmt(row['female_pct'])} | {fmt(row['internet_use_pct'])} | "
            f"{fmt(row['social_participation_pct'])} | {fmt(row['healthy_aging_pct'])} |"
        )

    lines += [
        "",
        "## 初步模型",
        "",
        "结局为探索性 `healthy_aging_binary`：自评健康、ADL、多病共存、抑郁、认知五类指标中，至少 4 类可用且健康比例达到 80%。模型为 person-wave 层面的 logistic GLM，控制年龄、性别、教育、婚姻、城乡、工作状态；pooled 模型额外控制 cohort 固定效应。标准误优先按个体聚类。",
        "",
        "### Internet use -> healthy ageing",
        "",
        "| Model | N | OR | 95% CI | P | Status |",
        "|---|---:|---:|---|---:|---|",
    ]
    for _, row in internet_models.iterrows():
        ci = f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))}" if row.get("status") == "ok" else ""
        lines.append(
            f"| {row.get('model')} | {int(row.get('n', 0))} | {fmt(row.get('odds_ratio'))} | {ci} | {fmt(row.get('p_value'), 4)} | {row.get('status')} |"
        )

    lines += [
        "",
        "### Social participation -> healthy ageing",
        "",
        "| Model | N | OR | 95% CI | P | Status |",
        "|---|---:|---:|---|---:|---|",
    ]
    for _, row in social_models.iterrows():
        ci = f"{fmt(row.get('ci_low'))}-{fmt(row.get('ci_high'))}" if row.get("status") == "ok" else ""
        lines.append(
            f"| {row.get('model')} | {int(row.get('n', 0))} | {fmt(row.get('odds_ratio'))} | {ci} | {fmt(row.get('p_value'), 4)} | {row.get('status')} |"
        )

    lines += [
        "",
        "## 重要解释",
        "",
        "- 这是第一版可行性分析，不是最终论文模型。",
        "- 当前模型使用 person-wave 数据，已聚类个体标准误，但仍应升级为严格纵向设计，例如滞后暴露、固定效应、轨迹模型或状态转移模型。",
        "- KLoSA 和 MHAS 的互联网变量缺失或不可比，互联网模型不会覆盖全部七国。",
        "- 各国 CESD 量表和认知量表不同，当前做了实用型 harmonization；正式投稿前需要在方法中逐一说明阈值和敏感性分析。",
        "- 下一步最值得做的是：健康老龄化轨迹、frailty/intrinsic capacity 转移、多队列异质性森林图。",
        "",
        "## 输出文件",
        "",
        "- `harmonized_variable_mapping.csv`：跨国变量映射。",
        "- `cohort_overview.csv`：队列层面描述统计。",
        "- `wave_counts.csv`：各队列各波次样本量。",
        "- `variable_availability.csv`：关键变量可用率。",
        "- `model_internet_healthy_aging.csv`：互联网使用模型。",
        "- `model_social_healthy_aging.csv`：社会参与模型。",
        "- `harmonized_core.parquet`：统一后的 person-wave 核心数据。",
    ]
    (output_dir / "analysis_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--skip-parquet", action="store_true", help="Do not write harmonized_core.parquet")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    harmonized, mapping = read_and_harmonize(args.data_dir, args.chunksize)
    harmonized = add_derived_variables(harmonized)

    overview, wave_counts, availability = build_descriptives(harmonized)
    internet_models = fit_models(harmonized, "internet_use")
    social_models = fit_models(harmonized, "social_participation")

    mapping.to_csv(args.output_dir / "harmonized_variable_mapping.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(args.output_dir / "cohort_overview.csv", index=False, encoding="utf-8-sig")
    wave_counts.to_csv(args.output_dir / "wave_counts.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(args.output_dir / "variable_availability.csv", index=False, encoding="utf-8-sig")
    internet_models.to_csv(args.output_dir / "model_internet_healthy_aging.csv", index=False, encoding="utf-8-sig")
    social_models.to_csv(args.output_dir / "model_social_healthy_aging.csv", index=False, encoding="utf-8-sig")
    if not args.skip_parquet:
        harmonized.to_parquet(args.output_dir / "harmonized_core.parquet", index=False)
    write_report(args.output_dir, overview, availability, internet_models, social_models)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
