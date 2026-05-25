"""
This script builds a pooled, per-model feature table from the evaluated JSONL
outputs in ``results/evaluation_results/`` and fits two interpretable models:

* logistic regression with L2 regularization for the primary inference
* random forest for a robustness-oriented feature ranking

The analysis is designed to answer: which question properties make a model
most likely to produce a confident-wrong response?

Outputs are written under ``results/diagnostic_analysis/``:

* ``feature_table.csv`` - one row per evaluated question/variant with features
* ``model_metrics.csv`` - per-model AUC and class balance summary
* ``logistic_coefficients.csv`` - coefficients, odds ratios, and bootstrap CIs
* ``random_forest_importance.csv`` - feature importance rankings
* ``cross_model_stability.csv`` - aggregate stability across models
* ``summary.json`` - compact machine-readable summary
* ``figures/`` - coefficient and importance plots

The analysis is intentionally restricted to processed inputs and low-risk
features: relation type, task type, anchor presence, question length, entity
count, and any precomputed rarity flag already present in ``data/``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_BOOTSTRAPS = 200
DEFAULT_CV_FOLDS = 5
BOOTSTRAP_ROW_CAP = 6000
ANALYSIS_SUBSETS = {"long_tail", "anchor_induced", "near_true", "head_tail_rarity", "counterfactual", "non_existant"}


LONGTAIL_RELATION_LABELS = {
    "P19": "birthplace",
    "P20": "place_of_death",
    "P27": "citizenship",
    "P30": "continent",
    "P36": "capital",
    "P37": "official_language",
    "P50": "author",
    "P57": "director",
    "P86": "composer",
    "P106": "occupation",
    "P123": "publisher",
    "P127": "owner",
    "P131": "administrative_region",
    "P136": "genre",
    "P175": "performer",
    "P495": "production_country",
}

LONGTAIL_RELATION_GROUPS = {
    "P19": "biographical",
    "P20": "biographical",
    "P27": "biographical",
    "P30": "geographic",
    "P36": "geographic",
    "P37": "organizational",
    "P50": "creative",
    "P57": "creative",
    "P86": "creative",
    "P106": "organizational",
    "P123": "creative",
    "P127": "organizational",
    "P131": "geographic",
    "P136": "creative",
    "P175": "creative",
    "P495": "creative",
}

SURFACE_PATTERNS = [
    (r"place of birth|born in", "birthplace"),
    (r"place of death|died in", "place_of_death"),
    (r"country of citizenship|citizenship", "citizenship"),
    (r"official language|language", "official_language"),
    (r"what is the capital", "capital"),
    (r"who is the author|written by|who wrote", "author"),
    (r"who directed|directed by", "director"),
    (r"who composed|music for", "composer"),
    (r"what is the occupation|professional background", "occupation"),
    (r"who published", "publisher"),
    (r"who owns", "owner"),
    (r"administrative region|located in", "location"),
    (r"what genre|specific conventions", "genre"),
    (r"who performed", "performer"),
    (r"produced in", "production_country"),
    (r"when was .* released|released in", "release_date"),
    (r"when was .* founded", "founding_date"),
    (r"educat", "education"),
    (r"win an emmy|win an oscar", "award_temporal"),
    (r"films stars|features", "compound_media_relation"),
]

SOURCE_GLOBS = {
    "long_tail": ["data/long_tailed.csv"],
    "anchor_induced": ["data/anchor_induced.csv"],
    "near_true": ["data/near_true.csv"],
    "head_tail_rarity": ["data/head-tail-rarity.csv"],
    "counterfactual": ["data/counterfactual.csv"],
    "non_existant": ["data/non-existant.csv"],
}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_for_match(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value)).strip()


def is_potential_entity(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if re.fullmatch(r"\d{4}(-\d{2}-\d{2})?", text):
        return False
    if re.fullmatch(r"\d+", text):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text))


def safe_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_log1p(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0:
        return None
    return float(math.log1p(value))


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_source_csvs(root: Path, globs: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for pattern in globs:
        for match in sorted(root.glob(pattern)):
            if match.exists():
                frames.append(pd.read_csv(match))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def infer_surface_form(question: str, fallback: str = "other") -> str:
    text = normalize_text(question)
    for pattern, label in SURFACE_PATTERNS:
        if re.search(pattern, text):
            return label
    return fallback


def relation_group_from_feature(feature: str) -> str:
    if feature in LONGTAIL_RELATION_GROUPS:
        return LONGTAIL_RELATION_GROUPS[feature]
    if feature in {"release_date", "founding_date", "award_temporal"}:
        return "temporal"
    if feature in {"education", "compound_media_relation"}:
        return "compositional"
    return feature


def cw_positive(outcome_label: str) -> bool:
    return outcome_label in {"confident_wrong", "confident_hallucinated"}


def mixed_variant_label(question_variant: str) -> str:
    return "anchor_induced" if question_variant in {"default", "grounded"} else "compositional_no_anchor"


def build_source_table(subset: str, root: Path) -> pd.DataFrame:
    if subset == "anchor_induced":
        df = load_source_csvs(root, SOURCE_GLOBS[subset])
        if df.empty:
            return df

        grounded = df.copy()
        grounded["question_variant"] = "default"
        grounded["question"] = grounded["grounded_question"]
        grounded["has_anchor"] = 1
        grounded["mixed_subcondition"] = "anchor_induced"
        grounded["relation_feature"] = "release_date"
        grounded["relation_group"] = "temporal"
        grounded["relation_surface"] = "release_date"

        ungrounded = df.copy()
        ungrounded["question_variant"] = "ungrounded"
        ungrounded["question"] = ungrounded["ungrounded_question"]
        ungrounded["has_anchor"] = 0
        ungrounded["mixed_subcondition"] = "compositional_no_anchor"
        ungrounded["relation_feature"] = "release_date"
        ungrounded["relation_group"] = "temporal"
        ungrounded["relation_surface"] = "release_date"

        out = pd.concat([grounded, ungrounded], ignore_index=True, sort=False)
        return out.drop(columns=[c for c in ["grounded_question", "ungrounded_question"] if c in out.columns])

    df = load_source_csvs(root, SOURCE_GLOBS.get(subset, []))
    if df.empty:
        return df

    df = df.copy()
    if "question_variant" not in df.columns:
        df["question_variant"] = "default"
    df["task_type"] = subset

    if subset == "long_tail":
        df["relation_feature"] = df["relation"].astype(str)
        df["relation_group"] = df["relation_feature"].map(lambda x: LONGTAIL_RELATION_GROUPS.get(x, "other"))
        df["relation_surface"] = df["relation_feature"].map(lambda x: LONGTAIL_RELATION_LABELS.get(x, "other"))
        df["has_anchor"] = 1
    elif subset in {"counterfactual", "non_existant", "head_tail_rarity"}:
        df["relation_surface"] = df["question"].map(lambda q: infer_surface_form(q, fallback="compound_media_relation"))
        df["relation_feature"] = df["relation_surface"]
        df["relation_group"] = df["relation_feature"].map(relation_group_from_feature)
        df["has_anchor"] = 1
    elif subset == "near_true":
        df["relation_feature"] = df["fact_type"].fillna(df["perturbation_type"]).astype(str)
        df["relation_surface"] = df["question"].map(lambda q: infer_surface_form(q, fallback="distractor_surface"))
        df["relation_group"] = df["perturbation_type"].fillna("unknown").astype(str)
        df["has_anchor"] = 1
    return df


def load_evaluation_records(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for subset_dir in sorted(results_dir.iterdir() if results_dir.exists() else []):
        if not subset_dir.is_dir():
            continue
        subset = subset_dir.name
        for jsonl_path in sorted(subset_dir.glob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("type") == "metadata":
                        continue

                    if subset == "anchor_induced" and isinstance(payload.get("eval"), dict) and "default" in payload.get("eval", {}):
                        for variant in ("default", "ungrounded"):
                            question_text = payload.get("ungrounded", {}).get("question") if variant == "ungrounded" else payload.get("question")
                            rows.append({
                                "model": payload.get("model") or payload.get("metadata", {}).get("model"),
                                "subset": subset,
                                "question_variant": variant,
                                "question": question_text,
                                "ground_truth": payload.get("ground_truth"),
                                "outcome_label": payload.get("eval", {}).get(variant, {}).get("outcome_label", ""),
                                "correctness": payload.get("eval", {}).get(variant, {}).get("correctness", ""),
                            })
                    else:
                        eval_obj = payload.get("eval", {}) if isinstance(payload.get("eval"), dict) else {}
                        rows.append({
                            "model": payload.get("model") or payload.get("metadata", {}).get("model"),
                            "subset": subset,
                            "question_variant": payload.get("question_variant", "default") or "default",
                            "question": payload.get("question"),
                            "ground_truth": payload.get("ground_truth"),
                            "outcome_label": eval_obj.get("outcome_label", ""),
                            "correctness": eval_obj.get("correctness", ""),
                        })

    return pd.DataFrame(rows)


def preferred_candidate_columns(subset: str, source_row: pd.Series, question: str) -> List[Tuple[str, object]]:
    candidates: List[Tuple[str, object]] = []
    question_norm = normalize_for_match(question)

    def add(role: str, value: object) -> None:
        candidate_norm = normalize_for_match(value)
        if is_potential_entity(value) and candidate_norm and (
            candidate_norm in question_norm or question_norm in candidate_norm
        ):
            candidates.append((role, value))

    if subset == "long_tail":
        add("entity", source_row.get("entity"))
    elif subset == "anchor_induced":
        add("head_entity", source_row.get("head_entity"))
        add("tail_entity", source_row.get("tail_entity"))
    elif subset in {"counterfactual", "non_existant", "head_tail_rarity"}:
        add("entity1", source_row.get("entity1"))
        add("entity2", source_row.get("entity2"))
    elif subset == "near_true":
        add("entity", source_row.get("entity"))
        add("wrong_value", source_row.get("wrong_value"))
    else:
        for column in source_row.index:
            if column in {"question", "question_variant", "model", "subset"}:
                continue
            add(column, source_row.get(column))

    seen = set()
    unique: List[Tuple[str, object]] = []
    for role, value in candidates:
        key = normalize_text(value)
        if key not in seen:
            unique.append((role, value))
            seen.add(key)
    return unique


def build_feature_row(
    subset: str,
    model: str,
    source_row: pd.Series,
    eval_row: pd.Series,
) -> Dict[str, object]:
    question = str(eval_row.get("question") or source_row.get("question") or "")
    question_variant = str(eval_row.get("question_variant") or source_row.get("question_variant") or "default")
    outcome_label = str(eval_row.get("outcome_label") or "")
    cw_label = int(cw_positive(outcome_label))

    candidates = preferred_candidate_columns(subset, source_row, question)
    entity_records: List[Dict[str, object]] = []
    for role, entity_label in candidates:
        entity_records.append(
            {
                "role": role,
                "entity": str(entity_label),
            }
        )

    anchor_present = int(bool(entity_records))
    if subset == "anchor_induced" and question_variant == "ungrounded":
        anchor_present = 0

    question_length = len(question.split()) if question.strip() else 0
    relation_feature = str(source_row.get("relation_feature") or source_row.get("fact_type") or source_row.get("perturbation_type") or source_row.get("relation") or infer_surface_form(question))
    relation_group = str(source_row.get("relation_group") or relation_group_from_feature(relation_feature))
    relation_surface = str(source_row.get("relation_surface") or infer_surface_form(question))
    primary_role = entity_records[0]["role"] if entity_records else None
    any_entity_rare = source_row.get("any_entity_rare") if "any_entity_rare" in source_row.index else None
    if pd.isna(any_entity_rare):
        page_views = safe_float(source_row.get("page_views"))
        head_sitelinks = safe_float(source_row.get("head_sitelinks"))
        tail_sitelinks = safe_float(source_row.get("tail_sitelinks"))
        if subset == "long_tail" and page_views is not None:
            any_entity_rare = int(page_views <= 1000)
        elif subset == "anchor_induced" and tail_sitelinks is not None:
            any_entity_rare = int(tail_sitelinks < 10)
        elif subset == "head_tail_rarity":
            any_entity_rare = 1
        elif subset in {"counterfactual", "non_existant"}:
            any_entity_rare = 0
        elif head_sitelinks is not None and tail_sitelinks is not None:
            any_entity_rare = int(min(head_sitelinks, tail_sitelinks) < 10)
    any_entity_rare = int(any_entity_rare) if pd.notna(any_entity_rare) else np.nan

    return {
        "model": model,
        "subset": subset,
        "question_variant": question_variant,
        "question": question,
        "cw_label": cw_label,
        "outcome_label": outcome_label,
        "correctness": str(eval_row.get("correctness") or ""),
        "task_type": subset,
        "has_anchor": anchor_present,
        "relation_feature": relation_feature,
        "relation_group": relation_group,
        "relation_surface": relation_surface,
        "question_length": question_length,
        "entity_count": len(entity_records),
        "candidate_entities": json.dumps(entity_records, ensure_ascii=False),
        "primary_entity_role": primary_role or "",
        "any_entity_rare": any_entity_rare,
        "all_entities": "|".join(x["entity"] for x in entity_records),
    }


def load_any_entity_rare_lookup(root: Path) -> pd.DataFrame:
    temp_feature_table = root / "results" / "diagnostic_analysis_temp" / "feature_table.csv"
    if not temp_feature_table.exists():
        return pd.DataFrame()

    temp_df = pd.read_csv(
        temp_feature_table,
        usecols=lambda col: col in {"subset", "question", "question_variant", "any_entity_rare"},
    )
    if temp_df.empty or "any_entity_rare" not in temp_df.columns:
        return pd.DataFrame()

    lookup_cols = ["subset", "question"]
    if "question_variant" in temp_df.columns:
        lookup_cols.append("question_variant")

    temp_df = temp_df[lookup_cols + ["any_entity_rare"]].dropna(subset=["subset", "question"])
    return temp_df.drop_duplicates(subset=lookup_cols, keep="first")


def build_feature_table(root: Path, results_dir: Path) -> pd.DataFrame:
    eval_df = load_evaluation_records(results_dir)
    if eval_df.empty:
        return eval_df

    eval_df = eval_df[eval_df["subset"].isin(ANALYSIS_SUBSETS)].copy()
    if eval_df.empty:
        return eval_df

    feature_rows: List[Dict[str, object]] = []
    rarity_lookup = load_any_entity_rare_lookup(root)
    for subset, subset_eval in eval_df.groupby("subset", sort=False):
        source_df = build_source_table(subset, root)
        if source_df.empty:
            continue

        key_cols = ["question", "question_variant"] if subset == "anchor_induced" else ["question"]
        merged = subset_eval.merge(source_df, on=key_cols, how="left", suffixes=("", "_source"))
        if not rarity_lookup.empty:
            rarity_keys = ["subset", "question"]
            if subset == "anchor_induced":
                rarity_keys.append("question_variant")
            merged = merged.merge(rarity_lookup, on=rarity_keys, how="left", suffixes=("", "_lookup"))
            if "any_entity_rare_lookup" in merged.columns:
                merged["any_entity_rare"] = merged["any_entity_rare"].combine_first(merged["any_entity_rare_lookup"])
                merged = merged.drop(columns=["any_entity_rare_lookup"])
        for _, row in merged.iterrows():
            feature_rows.append(build_feature_row(subset, str(row["model"]), row, row))

    return pd.DataFrame(feature_rows)


def build_preprocessor(df: pd.DataFrame, feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric_cols = [
        "has_anchor",
        "question_length",
        "entity_count",
        "any_entity_rare",
    ]
    numeric_cols = [c for c in numeric_cols if c in feature_columns]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def bootstrap_logistic_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    bootstrap_samples: int,
) -> Dict[str, np.ndarray]:
    if bootstrap_samples <= 0:
        return {}

    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    if len(positives) == 0 or len(negatives) == 0:
        return {}

    rng = np.random.default_rng(42)
    samples: Dict[str, List[float]] = {feature: [] for feature in feature_names}

    for _ in range(bootstrap_samples):
        pos_indices = rng.choice(positives, size=len(positives), replace=True)
        neg_indices = rng.choice(negatives, size=len(negatives), replace=True)
        boot_indices = np.concatenate([pos_indices, neg_indices])
        if len(boot_indices) > BOOTSTRAP_ROW_CAP:
            boot_indices = rng.choice(boot_indices, size=BOOTSTRAP_ROW_CAP, replace=False)
        rng.shuffle(boot_indices)
        y_boot = y[boot_indices]
        if len(np.unique(y_boot)) < 2:
            continue
        X_boot = X[boot_indices]
        try:
            model = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
            model.fit(X_boot, y_boot)
            for idx, feature in enumerate(feature_names):
                samples[feature].append(float(model.coef_.ravel()[idx]))
        except Exception:
            continue

    return {feature: np.asarray(values, dtype=float) for feature, values in samples.items() if values}


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def save_model_plots(model_name: str, coef_df: pd.DataFrame, rf_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    coef_plot = coef_df.copy()
    coef_plot = coef_plot.assign(abs_coef=coef_plot["coef"].abs()).sort_values("abs_coef", ascending=False).head(15)
    if not coef_plot.empty:
        plt.figure(figsize=(10, 7))
        colors = ["#c94c4c" if value < 0 else "#2f7ed8" for value in coef_plot["coef"]]
        lower_err = np.maximum(coef_plot["coef"] - coef_plot["ci_low"], 0)
        upper_err = np.maximum(coef_plot["ci_high"] - coef_plot["coef"], 0)
        yerr = np.vstack([lower_err, upper_err])
        plt.barh(coef_plot["feature"], coef_plot["coef"], xerr=yerr, color=colors, alpha=0.9)
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Log-odds coefficient")
        plt.title(f"{model_name}: logistic coefficients")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{sanitize_filename(model_name)}_logistic_coefficients.png", dpi=200)
        plt.close()

    rf_plot = rf_df.head(15)
    if not rf_plot.empty:
        plt.figure(figsize=(10, 7))
        plt.barh(rf_plot["feature"], rf_plot["importance"], color="#4c8c4a", alpha=0.9)
        plt.gca().invert_yaxis()
        plt.xlabel("Random forest importance")
        plt.title(f"{model_name}: feature importance")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{sanitize_filename(model_name)}_rf_importance.png", dpi=200)
        plt.close()


def aggregate_feature_stability(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return coef_df

    rows = []
    for feature, group in coef_df.groupby("feature"):
        coefs = group["coef"].astype(float).to_numpy()
        rows.append(
            {
                "feature": feature,
                "models": int(group["model"].nunique()),
                "mean_coef": float(np.mean(coefs)),
                "median_coef": float(np.median(coefs)),
                "mean_odds_ratio": float(np.mean(np.exp(coefs))),
                "positive_rate": float(np.mean(coefs > 0)),
                "negative_rate": float(np.mean(coefs < 0)),
                "sign_consistency": float(max(np.mean(coefs > 0), np.mean(coefs < 0))),
                "ci_excludes_zero_rate": float(np.mean((group["ci_low"] > 0) | (group["ci_high"] < 0))),
            }
        )
    return pd.DataFrame(rows).sort_values(["sign_consistency", "models", "mean_coef"], ascending=[False, False, False])


def write_summary_plots(stability_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = stability_df.head(15)
    if top.empty:
        return
    plt.figure(figsize=(11, 7))
    plt.barh(top["feature"], top["mean_coef"], color=["#2f7ed8" if x >= 0 else "#c94c4c" for x in top["mean_coef"]])
    plt.axvline(0, color="black", linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean log-odds coefficient across models")
    plt.title("Cross-model coefficient stability")
    plt.tight_layout()
    plt.savefig(fig_dir / "cross_model_coefficient_stability.png", dpi=200)
    plt.close()


def fit_logistic_and_rf(df: pd.DataFrame, model_name: str, bootstrap_samples: int, cv_folds: int, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    base_features = [
        "relation_feature",
        "relation_group",
        "has_anchor",
        "question_length",
        "entity_count",
        "any_entity_rare",
        "task_type",
    ]
    model_df = df[df["model"] == model_name].copy()
    if "any_entity_rare" in model_df.columns and model_df["any_entity_rare"].isna().all():
        base_features = [feature for feature in base_features if feature != "any_entity_rare"]
    model_df = model_df[base_features + ["cw_label", "subset", "outcome_label"]].copy()
    model_df = model_df.dropna(subset=["cw_label"])

    class_counts = Counter(model_df["cw_label"])
    positive_count = int(class_counts.get(1, 0))
    negative_count = int(class_counts.get(0, 0))
    metrics: Dict[str, object] = {
        "model": model_name,
        "n_rows": int(len(model_df)),
        "positives": positive_count,
        "negatives": negative_count,
        "positive_rate": float(model_df["cw_label"].mean()) if len(model_df) else 0.0,
        "notes": "",
    }

    if len(model_df) < 30 or positive_count < 5 or negative_count < 5:
        metrics["notes"] = "insufficient_class_balance"
        return pd.DataFrame(), pd.DataFrame(), metrics

    feature_cols = [c for c in base_features if c in model_df.columns]
    preprocessor = build_preprocessor(model_df, feature_cols)
    X = preprocessor.fit_transform(model_df[feature_cols])
    feature_names = list(preprocessor.get_feature_names_out())
    y = model_df["cw_label"].astype(int).to_numpy()

    logistic = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    logistic.fit(X, y)

    boot = bootstrap_logistic_coefficients(X, y, feature_names, bootstrap_samples)
    coef_rows = []
    for idx, feature in enumerate(feature_names):
        samples = boot.get(feature, np.array([]))
        coef_value = float(logistic.coef_.ravel()[idx])
        coef_rows.append(
            {
                "model": model_name,
                "feature": feature,
                "coef": coef_value,
                "odds_ratio": float(np.exp(coef_value)),
                "ci_low": float(np.percentile(samples, 2.5)) if len(samples) else np.nan,
                "ci_high": float(np.percentile(samples, 97.5)) if len(samples) else np.nan,
                "odds_ratio_low": float(np.exp(np.percentile(samples, 2.5))) if len(samples) else np.nan,
                "odds_ratio_high": float(np.exp(np.percentile(samples, 97.5))) if len(samples) else np.nan,
                "bootstrap_n": int(len(samples)),
            }
        )
    coef_df = pd.DataFrame(coef_rows)

    cv_splits = min(cv_folds, max(2, min(positive_count, negative_count)))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(model_df, feature_cols)),
            ("logreg", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=42)),
        ]
    )
    auc_scores = cross_val_score(pipe, model_df[feature_cols], y, cv=cv, scoring="roc_auc")
    metrics["auc_mean"] = float(np.mean(auc_scores))
    metrics["auc_std"] = float(np.std(auc_scores))
    metrics["auc_scores"] = auc_scores.tolist()

    rf_pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(model_df, feature_cols)),
            (
                "rf",
                RandomForestClassifier(
                        n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    min_samples_leaf=3,
                ),
            ),
        ]
    )
    rf_pipe.fit(model_df[feature_cols], y)
    rf_features = list(rf_pipe.named_steps["preprocess"].get_feature_names_out())
    rf_importances = rf_pipe.named_steps["rf"].feature_importances_
    rf_df = pd.DataFrame({"model": model_name, "feature": rf_features, "importance": rf_importances}).sort_values("importance", ascending=False).reset_index(drop=True)

    save_model_plots(model_name, coef_df, rf_df, out_dir)
    return coef_df, rf_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANDOR diagnostic boundary analysis.")
    parser.add_argument("--results-dir", default="results/evaluation_results", help="Directory with evaluated JSONL outputs.")
    parser.add_argument("--output-dir", default="results/diagnostic_analysis", help="Directory for analysis outputs.")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAPS, help="Bootstrap resamples for coefficient CIs.")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="Max cross-validation folds for AUC.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names to analyze.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = (root / args.results_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table = build_feature_table(root, results_dir)

    if feature_table.empty:
        raise SystemExit("No evaluation records found. Check results/evaluation_results for evaluated JSONL outputs.")

    if args.models:
        feature_table = feature_table[feature_table["model"].isin(set(args.models))].copy()

    feature_table.to_csv(output_dir / "feature_table.csv", index=False)

    models = sorted(feature_table["model"].dropna().unique().tolist())
    if not models:
        raise SystemExit("No models found after filtering.")

    all_coef_frames: List[pd.DataFrame] = []
    all_rf_frames: List[pd.DataFrame] = []
    model_metric_rows: List[Dict[str, object]] = []
    model_notes: Dict[str, str] = {}

    for model_name in models:
        coef_df, rf_df, metrics = fit_logistic_and_rf(feature_table, model_name, args.bootstrap_samples, args.cv_folds, output_dir)
        model_metric_rows.append(metrics)
        model_notes[model_name] = str(metrics.get("notes", ""))
        if not coef_df.empty:
            all_coef_frames.append(coef_df)
        if not rf_df.empty:
            all_rf_frames.append(rf_df)

    model_metrics_df = pd.DataFrame(model_metric_rows)
    model_metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    coef_all = pd.concat(all_coef_frames, ignore_index=True) if all_coef_frames else pd.DataFrame()
    rf_all = pd.concat(all_rf_frames, ignore_index=True) if all_rf_frames else pd.DataFrame()
    if not coef_all.empty:
        coef_all.to_csv(output_dir / "logistic_coefficients.csv", index=False)
        stability_df = aggregate_feature_stability(coef_all)
        stability_df.to_csv(output_dir / "cross_model_stability.csv", index=False)
        write_summary_plots(stability_df, output_dir)
    else:
        stability_df = pd.DataFrame()

    if not rf_all.empty:
        rf_all.to_csv(output_dir / "random_forest_importance.csv", index=False)

    summary = {
        "n_rows": int(len(feature_table)),
        "models": models,
        "model_notes": model_notes,
        "feature_columns": [c for c in feature_table.columns if c not in {"model", "subset", "question", "question_variant", "cw_label", "outcome_label", "correctness", "candidate_entities", "all_entities", "entity_lookup_sources"}],
        "output_dir": str(output_dir),
        "tasks": sorted(feature_table["subset"].dropna().unique().tolist()),
        "cw_rate": float(feature_table["cw_label"].mean()),
        "class_balance": {str(key): int(value) for key, value in feature_table["cw_label"].value_counts().to_dict().items()},
    }
    if not model_metrics_df.empty:
        summary["auc_mean_across_models"] = float(model_metrics_df["auc_mean"].dropna().mean())
        summary["auc_median_across_models"] = float(model_metrics_df["auc_mean"].dropna().median())
    if not stability_df.empty:
        summary["top_features_by_stability"] = [
            {
                "feature": str(row["feature"]),
                "mean_coef": float(row["mean_coef"]),
                "positive_rate": float(row["positive_rate"]),
                "sign_consistency": float(row["sign_consistency"]),
            }
            for _, row in stability_df.head(10)[["feature", "mean_coef", "positive_rate", "sign_consistency"]].iterrows()
        ]

    write_json(output_dir / "summary.json", summary)
    print(f"Saved diagnostic analysis to {output_dir}")


if __name__ == "__main__":
    main()