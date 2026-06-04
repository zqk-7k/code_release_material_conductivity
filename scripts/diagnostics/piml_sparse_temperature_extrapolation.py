"""Sparse low-temperature extrapolation diagnostics for PIML versus DNN.

This script keeps the locked PIML and DNN configurations selected in the
grouped campaign. It does not choose a model from the results produced here.
The diagnostic asks whether the Arrhenius-constrained model helps when each
material/process/source group contributes only a small number of low-temperature
records before extrapolation to high temperature.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "windows_material_conductivity_training_handoff" / "windows_experiments" / "piml_metric_optimization"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_physics_enhanced_campaign as campaign  # noqa: E402
import run_review_validation_experiments as review  # noqa: E402

TEMP = campaign.TEMP
TARGET = campaign.TARGET
CELSIUS_OFFSET = 273.15
FEATURE_MODE = "tfidf16"


def flatten_text(values: np.ndarray) -> np.ndarray:
    return values.squeeze()


def to_dense(values):
    return values.toarray() if hasattr(values, "toarray") else values


def onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def stable_feature_pipeline() -> ColumnTransformer:
    numeric = [
        "total_dopant_fraction",
        "average_dopant_radius",
        "average_dopant_valence",
        "number_of_dopants",
        "maximum_sintering_temperature",
        "total_sintering_duration",
    ]
    categorical = ["synthesis_method", "primary_dopant_element"]
    transformers = [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", onehot())]),
                categorical,
            ),
    ]
    if FEATURE_MODE == "tfidf16":
        transformers.append(
                (
                "text",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", FunctionTransformer(flatten_text, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=16, stop_words="english")),
                        ("dense", FunctionTransformer(to_dense, accept_sparse=True)),
                    ]
                ),
                ["material_source_and_purity"],
            ),
        )
    elif FEATURE_MODE != "no_text":
        raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def stable_empirical_ea_targets(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    groups = campaign.group_series(train)
    targets = np.full(len(train), np.nan, dtype=np.float32)
    group_values = []
    for _, positions in train.groupby(groups).groups.items():
        frame = train.loc[positions]
        if frame[TEMP].nunique() < 2:
            continue
        x = 1.0 / frame[TEMP].to_numpy(dtype=float)
        y = frame[TARGET].to_numpy(dtype=float) + np.log10(frame[TEMP].to_numpy(dtype=float))
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered * x_centered))
        if denom <= 1e-18:
            continue
        slope = float(np.sum(x_centered * (y - y.mean())) / denom)
        ea = -slope * campaign.KB_EV * np.log(10.0)
        if 0.03 <= ea <= 2.0:
            local_indices = train.index.get_indexer(positions)
            targets[local_indices] = ea
            group_values.append(float(ea))
    mask = ~np.isnan(targets)
    details = {
        "n_rows_with_empirical_ea": int(mask.sum()),
        "n_groups_with_empirical_ea": len(group_values),
        "median_empirical_ea": float(np.median(group_values)) if group_values else None,
    }
    return np.nan_to_num(targets, nan=0.0), mask.astype(np.float32), details


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def paired_bootstrap_delta(
    y: np.ndarray,
    piml: np.ndarray,
    dnn: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 20260603,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_rmse = np.sqrt(mean_squared_error(y[idx], piml[idx]))
        d_rmse = np.sqrt(mean_squared_error(y[idx], dnn[idx]))
        values[i] = d_rmse - p_rmse
    return {
        "delta_rmse_ci95_low": float(np.quantile(values, 0.025)),
        "delta_rmse_ci95_high": float(np.quantile(values, 0.975)),
        "prob_delta_le_0": float(np.mean(values <= 0)),
    }


def select_sparse_low_records(low: pd.DataFrame, mode: str) -> pd.DataFrame:
    low = low.copy()
    low["_group"] = campaign.group_series(low).to_numpy()
    if mode == "full_low":
        return low.drop(columns=["_group"])
    if mode == "top1_low_per_group":
        selected = low.sort_values(TEMP, ascending=False).groupby("_group", as_index=False).head(1)
        return selected.drop(columns=["_group"])
    if mode == "top2_low_per_group":
        selected = low.sort_values(TEMP, ascending=False).groupby("_group", as_index=False).head(2)
        return selected.drop(columns=["_group"])
    if mode == "random1_low_per_group":
        selected = low.groupby("_group", group_keys=False).sample(n=1, random_state=20260603)
        return selected.drop(columns=["_group"])
    raise ValueError(f"Unknown sparse mode: {mode}")


def train_ensemble(
    family: str,
    cfg: campaign.Config,
    fixed_epochs: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: list[int],
    output: Path,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    original_builder = campaign.base.build_serializable_feature_pipeline
    original_ea = campaign.empirical_ea_targets
    campaign.base.build_serializable_feature_pipeline = stable_feature_pipeline
    campaign.empirical_ea_targets = stable_empirical_ea_targets
    try:
        for seed in seeds:
            _, pred = campaign.train_one(
                cfg,
                train,
                None,
                {"matched_high_temperature": test},
                seed,
                fixed_epochs,
                0,
                output / family / f"seed_{seed}",
                device,
                fixed_epochs=fixed_epochs,
            )
            predictions.append(pred["matched_high_temperature"])
    finally:
        campaign.base.build_serializable_feature_pipeline = original_builder
        campaign.empirical_ea_targets = original_ea
    return np.mean(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-summary",
        default=str(EXP / "results" / "physics_enhanced_campaign_full" / "campaign_summary.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper_write" / "conductivity" / "results" / "piml_advantage_search" / "sparse_temperature_extrapolation"),
    )
    parser.add_argument("--seeds", default="0,1,2,42,2026")
    parser.add_argument("--modes", default="top1_low_per_group,top2_low_per_group,random1_low_per_group")
    parser.add_argument("--feature-mode", choices=["tfidf16", "no_text"], default="tfidf16")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    global FEATURE_MODE
    FEATURE_MODE = args.feature_mode
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]

    df = campaign.base.MaterialDataProcessor().load_and_preprocess_data_for_training_piml()
    temperature_c = df[TEMP] - CELSIUS_OFFSET
    low = df[temperature_c <= 800.0 + 1e-6].copy()
    high = df[temperature_c >= 900.0 - 1e-6].copy()
    low_groups = set(campaign.group_series(low).tolist())
    high_groups = campaign.group_series(high)
    matched_high = high[high_groups.isin(low_groups)].copy()

    models, epochs = review.selected_models(Path(args.campaign_summary))
    rows = []
    run_manifest = {
        "campaign_summary": str(Path(args.campaign_summary)),
        "seeds": seeds,
        "modes": modes,
        "device": str(device),
        "feature_mode": FEATURE_MODE,
        "n_low_full": int(len(low)),
        "n_matched_high": int(len(matched_high)),
        "model_configs": {family: asdict(cfg) for family, cfg in models.items()},
        "fixed_epochs": epochs,
        "protocol": "Train fixed selected architectures on sparse low-temperature records, then evaluate on the same matched high-temperature set.",
    }
    (output / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    y = matched_high[TARGET].to_numpy(float)
    for mode in modes:
        train = select_sparse_low_records(low, mode)
        preds = {}
        for family in ("piml", "dnn"):
            preds[family] = train_ensemble(
                family,
                models[family],
                epochs[family],
                train,
                matched_high,
                seeds,
                output / mode,
                device,
            )
        p = metrics(y, preds["piml"])
        d = metrics(y, preds["dnn"])
        boot = paired_bootstrap_delta(y, preds["piml"], preds["dnn"], args.bootstrap)
        row = {
            "mode": mode,
            "n_train_rows": int(len(train)),
            "n_train_groups": int(campaign.group_series(train).nunique()),
            "n_test_rows": int(len(matched_high)),
            "piml_rmse": p["rmse"],
            "dnn_rmse": d["rmse"],
            "delta_rmse_dnn_minus_piml": d["rmse"] - p["rmse"],
            "piml_mae": p["mae"],
            "dnn_mae": d["mae"],
            "delta_mae_dnn_minus_piml": d["mae"] - p["mae"],
            "piml_r2": p["r2"],
            "dnn_r2": d["r2"],
            **boot,
        }
        rows.append(row)
        pred_table = matched_high[["sample_id", TEMP, TARGET]].copy()
        pred_table["piml_prediction"] = preds["piml"]
        pred_table["dnn_prediction"] = preds["dnn"]
        pred_table.to_csv(output / f"{mode}_matched_high_predictions.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "sparse_temperature_extrapolation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
