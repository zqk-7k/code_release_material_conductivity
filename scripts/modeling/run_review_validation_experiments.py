"""Reviewer-facing diagnostics for the selected Arrhenius PIML and tuned DNN.

The configurations are read from the completed grouped screening campaign.
No model choice is made from the outputs produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_physics_enhanced_campaign as campaign  # noqa: E402

TEMP = campaign.TEMP
TARGET = campaign.TARGET
CELSIUS_OFFSET = 273.15
FEATURE_SETS = ("full", "no_text", "no_processing", "composition_only")


def feature_pipeline(feature_set: str) -> ColumnTransformer:
    composition_numeric = [
        "total_dopant_fraction",
        "average_dopant_radius",
        "average_dopant_valence",
        "number_of_dopants",
    ]
    processing_numeric = ["maximum_sintering_temperature", "total_sintering_duration"]
    transformers = [
        (
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
            composition_numeric + (processing_numeric if feature_set in {"full", "no_text"} else []),
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            ["primary_dopant_element"] + (["synthesis_method"] if feature_set in {"full", "no_text"} else []),
        ),
    ]
    if feature_set in {"full", "no_processing"}:
        transformers.append(
            (
                "text",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", FunctionTransformer(campaign.base.flatten_text_column, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
                        ("svd", TruncatedSVD(n_components=16, random_state=42)),
                    ]
                ),
                ["material_source_and_purity"],
            )
        )
    return ColumnTransformer(transformers=transformers)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


def selected_models(summary_path: Path) -> tuple[dict[str, campaign.Config], dict[str, int]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = {}
    epochs = {}
    for family in ("piml", "dnn"):
        raw = summary["selected"][family]
        models[family] = campaign.Config(**raw["config"])
        epochs[family] = int(raw["fixed_epochs"])
    return models, epochs


def train_fixed_ensemble(
    cfg: campaign.Config,
    fixed_epochs: int,
    train: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
    feature_set: str,
    seeds: list[int],
    output: Path,
    device: torch.device,
) -> dict:
    original_builder = campaign.base.build_serializable_feature_pipeline
    campaign.base.build_serializable_feature_pipeline = lambda: feature_pipeline(feature_set)
    predictions = {name: [] for name in tests}
    members = []
    try:
        for seed in seeds:
            result, prediction = campaign.train_one(
                cfg,
                train,
                None,
                tests,
                seed,
                fixed_epochs,
                0,
                output / cfg.family / feature_set / f"seed_{seed}",
                device,
                fixed_epochs=fixed_epochs,
            )
            members.append(result)
            for name in tests:
                predictions[name].append(prediction[name])
    finally:
        campaign.base.build_serializable_feature_pipeline = original_builder
    ensemble_metrics = {}
    for name, frame in tests.items():
        pred = np.mean(predictions[name], axis=0)
        ensemble_metrics[name] = metric(frame[TARGET].to_numpy(), pred)
        table = frame[["sample_id", TEMP, TARGET]].copy()
        table["prediction"] = pred
        table.to_csv(output / f"{cfg.family}_{feature_set}_{name}_predictions.csv", index=False)
    return {
        "config": asdict(cfg),
        "fixed_epochs": fixed_epochs,
        "feature_set": feature_set,
        "seeds": seeds,
        "metrics": ensemble_metrics,
        "members": members,
    }


def temperature_extrapolation(
    df: pd.DataFrame,
    models: dict[str, campaign.Config],
    epochs: dict[str, int],
    seeds: list[int],
    output: Path,
    device: torch.device,
) -> dict:
    temperature_c = df[TEMP] - CELSIUS_OFFSET
    low = df[temperature_c <= 800.0 + 1e-6].copy()
    high = df[temperature_c >= 900.0 - 1e-6].copy()
    low_groups = set(campaign.group_series(low).tolist())
    high_groups = campaign.group_series(high)
    high_seen_chemistry = high[high_groups.isin(low_groups)].copy()
    tests = {"all_high_temperature": high, "matched_material_high_temperature": high_seen_chemistry}
    results = {}
    for family, cfg in models.items():
        results[family] = train_fixed_ensemble(
            cfg, epochs[family], low, tests, "full", seeds, output / "temperature_extrapolation", device
        )
    comparison = {}
    for name in tests:
        p = results["piml"]["metrics"][name]
        d = results["dnn"]["metrics"][name]
        comparison[name] = {
            "n_rows": len(tests[name]),
            "piml": p,
            "dnn": d,
            "piml_beats_dnn": p["rmse"] < d["rmse"] and p["r2"] > d["r2"],
        }
    return {
        "protocol": "Train only at measurement temperatures <=800 C; evaluate without refitting at >=900 C.",
        "n_training_rows": len(low),
        "n_all_high_rows": len(high),
        "n_matched_material_high_rows": len(high_seen_chemistry),
        "comparison": comparison,
    }


def feature_ablation(
    df: pd.DataFrame,
    models: dict[str, campaign.Config],
    epochs: dict[str, int],
    seeds: list[int],
    output: Path,
    device: torch.device,
) -> list[dict]:
    parts = campaign.reserve_partitions(df)
    test = pd.concat([parts["confirmation_a"], parts["confirmation_b"]], ignore_index=True)
    rows = []
    for feature_set in FEATURE_SETS:
        for family, cfg in models.items():
            result = train_fixed_ensemble(
                cfg, epochs[family], parts["development"], {"pooled_confirmation": test},
                feature_set, seeds, output / "feature_ablation", device
            )
            score = result["metrics"]["pooled_confirmation"]
            rows.append({"model": family.upper(), "feature_set": feature_set, **score})
    pd.DataFrame(rows).to_csv(output / "feature_ablation_summary.csv", index=False)
    return rows


def load_final_models(
    family: str,
    cfg: campaign.Config,
    campaign_output: Path,
    seeds: list[int],
    device: torch.device,
) -> list[tuple[torch.nn.Module, object]]:
    loaded = []
    for seed in seeds:
        root = campaign_output / "final_models" / family / f"seed_{seed}"
        checkpoint = torch.load(root / "model.pth", map_location=device, weights_only=False)
        pipeline = joblib.load(root / "preprocessor.joblib")
        model = campaign.EnhancedPIML(checkpoint["input_dim"], cfg) if family == "piml" else campaign.TunedDNN(checkpoint["input_dim"], cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        loaded.append((model, pipeline))
    return loaded


def physics_consistency(
    df: pd.DataFrame,
    models: dict[str, campaign.Config],
    campaign_output: Path,
    seeds: list[int],
    output: Path,
    device: torch.device,
) -> dict:
    parts = campaign.reserve_partitions(df)
    dev = parts["development"]
    confirmation = pd.concat([parts["confirmation_a"], parts["confirmation_b"]], ignore_index=True)
    confirmation["_group"] = campaign.group_series(confirmation).to_numpy()
    templates = confirmation.drop_duplicates("_group").copy()
    grid_k = np.arange(500, 1001, 50, dtype=float) + CELSIUS_OFFSET
    dnn_mean = float(dev[TEMP].mean())
    dnn_scale = float(dev[TEMP].std(ddof=0))
    loaded = {
        family: load_final_models(family, models[family], campaign_output, seeds, device)
        for family in ("piml", "dnn")
    }
    curves = []
    for _, template in templates.iterrows():
        grid = pd.DataFrame([template.drop(labels=["_group"]).to_dict()] * len(grid_k))
        grid[TEMP] = grid_k
        for family in ("piml", "dnn"):
            predictions = []
            for model, pipeline in loaded[family]:
                x = torch.as_tensor(pipeline.transform(grid), dtype=torch.float32, device=device)
                with torch.no_grad():
                    if family == "piml":
                        temperature = torch.as_tensor(grid_k, dtype=torch.float32, device=device).view(-1, 1)
                        pred, _, _ = model(x, temperature)
                    else:
                        temperature = torch.as_tensor((grid_k - dnn_mean) / dnn_scale, dtype=torch.float32, device=device).view(-1, 1)
                        pred = model(x, temperature)
                predictions.append(pred.cpu().numpy().ravel())
            predicted = np.mean(predictions, axis=0)
            corrected = predicted + np.log10(grid_k)
            fitted = np.polyval(np.polyfit(1.0 / grid_k, corrected, 1), 1.0 / grid_k)
            arrhenius_r2 = r2_score(corrected, fitted) if np.var(corrected) > 1e-12 else 1.0
            violations = int(np.sum(np.diff(predicted) < -1e-8))
            curves.append(
                {
                    "group_id": template["_group"],
                    "model": family.upper(),
                    "monotonicity_violations": violations,
                    "arrhenius_linearity_r2": float(arrhenius_r2),
                }
            )
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(output / "physical_consistency_curves.csv", index=False)
    summary = {}
    for family, sub in curve_frame.groupby("model"):
        summary[family] = {
            "n_confirmation_material_groups": int(len(sub)),
            "n_curves_with_nonmonotonic_step": int((sub["monotonicity_violations"] > 0).sum()),
            "fraction_monotonic": float((sub["monotonicity_violations"] == 0).mean()),
            "median_arrhenius_linearity_r2": float(sub["arrhenius_linearity_r2"].median()),
            "minimum_arrhenius_linearity_r2": float(sub["arrhenius_linearity_r2"].min()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2,42,2026")
    parser.add_argument("--ablation-seeds", default="0,1,2")
    args = parser.parse_args()
    summary_path = Path(args.campaign_summary)
    campaign_output = summary_path.parent
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = campaign.base.MaterialDataProcessor().load_and_preprocess_data_for_training_piml()
    models, epochs = selected_models(summary_path)
    seeds = [int(value) for value in args.seeds.split(",")]
    ablation_seeds = [int(value) for value in args.ablation_seeds.split(",")]
    results = {
        "locked_selected_models": {family: {"config": asdict(cfg), "epochs": epochs[family]} for family, cfg in models.items()},
        "temperature_extrapolation": temperature_extrapolation(df, models, epochs, seeds, output, device),
        "feature_ablation": feature_ablation(df, models, epochs, ablation_seeds, output, device),
        "physical_consistency": physics_consistency(df, models, campaign_output, seeds, output, device),
    }
    (output / "review_validation_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
