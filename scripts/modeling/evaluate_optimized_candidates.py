"""Evaluate paper-relevant candidates and physical outputs of optimized PIML models."""

from __future__ import annotations

import argparse
import __main__
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_piml_optimization as opt  # noqa: E402

DOPANTS = {"Sc": 87.0, "Mg": 89.0}
VALENCE = {"Sc": 3.0, "Mg": 2.0}
CANDIDATES = [
    {"candidate": "paper_main", "d1": "Sc", "f1": 7.50, "d2": "Mg", "f2": 3.19, "sintering": 1505.0},
    {"candidate": "current_retrained_ga", "d1": "Sc", "f1": 6.27, "d2": "Mg", "f2": 4.97, "sintering": 1505.0},
    {"candidate": "de_follow_up", "d1": "Mg", "f1": 2.452, "d2": "Sc", "f2": 9.657, "sintering": 1648.6},
    {"candidate": "integer_cell_follow_up", "d1": "Sc", "f1": 8.0, "d2": "Mg", "f2": 3.0, "sintering": 1505.0},
]


def candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duration = float(df["total_sintering_duration"].median())
    for candidate in CANDIDATES:
        row = df.iloc[0].copy()
        f1, f2 = candidate["f1"] / 100.0, candidate["f2"] / 100.0
        total = f1 + f2
        row["sample_id"] = candidate["candidate"]
        row["material_source_and_purity"] = "AI Discovery Co-Doping"
        row["synthesis_method"] = "Solid State Reaction"
        row["temperature_kelvin"] = 1073.15
        row["operating_temperature"] = 800.0
        row["total_dopant_fraction"] = total
        row["average_dopant_radius"] = (DOPANTS[candidate["d1"]] * f1 + DOPANTS[candidate["d2"]] * f2) / total
        row["average_dopant_valence"] = (VALENCE[candidate["d1"]] * f1 + VALENCE[candidate["d2"]] * f2) / total
        row["number_of_dopants"] = 2
        row["primary_dopant_element"] = candidate["d1"] if f1 >= f2 else candidate["d2"]
        row["maximum_sintering_temperature"] = candidate["sintering"]
        row["total_sintering_duration"] = duration
        rows.append(row)
    return pd.DataFrame(rows)


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = opt.ModelConfig(**payload["config"])
    model = opt.create_piml(int(payload["input_dim"]), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def evaluate_model(name: str, checkpoint: Path, pipeline, candidates: pd.DataFrame, device: torch.device):
    model, cfg = load_model(checkpoint, device)
    x = pipeline.transform(candidates)
    with torch.no_grad():
        pred, ea, loga = model(
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(candidates["temperature_kelvin"].to_numpy(), dtype=torch.float32, device=device).view(-1, 1),
        )
    rows = []
    for idx, candidate in enumerate(CANDIDATES):
        rows.append(
            {
                "model": name,
                "candidate": candidate["candidate"],
                "dopant_1": candidate["d1"],
                "dopant_1_mol_percent": candidate["f1"],
                "dopant_2": candidate["d2"],
                "dopant_2_mol_percent": candidate["f2"],
                "sintering_temperature_C": candidate["sintering"],
                "predicted_log10_sigma": float(pred[idx].item()),
                "predicted_Ea_eV": float(ea[idx].item()),
                "predicted_log10A": float(loga[idx].item()),
                "checkpoint_sha256": opt.sha256_file(checkpoint),
                "config": cfg.name,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "tuning_summary.json").read_text(encoding="utf-8"))
    # Older runs serialized this public transformer while the trainer was launched as __main__.
    __main__.flatten_text_column = opt.flatten_text_column
    pipeline = joblib.load(run_dir / "tuning_preprocessor.joblib")
    df = opt.MaterialDataProcessor().load_and_preprocess_data_for_training_piml()
    candidates = candidate_frame(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    best_path = run_dir / summary["best_tuning_piml"]["checkpoint"]
    rows.extend(evaluate_model("best_single", best_path, pipeline, candidates, device))
    for member in summary["tuning_ensemble"]["members"]:
        config, seed = member.rsplit("__seed_", maxsplit=1)
        checkpoint = run_dir / "checkpoints" / f"{config}_seed_{seed}" / "best_piml_model.pth"
        rows.extend(evaluate_model(member, checkpoint, pipeline, candidates, device))
    results = pd.DataFrame(rows)
    ensemble = (
        results[results["model"].isin(summary["tuning_ensemble"]["members"])]
        .groupby("candidate", as_index=False)[["predicted_log10_sigma", "predicted_Ea_eV", "predicted_log10A"]]
        .mean()
    )
    ensemble["model"] = "ensemble_mean"
    lookup = pd.DataFrame(CANDIDATES).rename(
        columns={"d1": "dopant_1", "f1": "dopant_1_mol_percent", "d2": "dopant_2", "f2": "dopant_2_mol_percent", "sintering": "sintering_temperature_C"}
    )
    ensemble = ensemble.merge(lookup, on="candidate", how="left")
    results = pd.concat([results, ensemble], ignore_index=True, sort=False)
    results.to_csv(run_dir / "optimized_candidate_sensitivity.csv", index=False)
    best_eval = pd.read_csv(best_path.parent / "evaluation_predictions.csv")
    physics = {
        "run_dir": str(run_dir),
        "best_model": summary["best_tuning_piml"]["model"],
        "best_model_metrics": {"rmse": summary["best_tuning_piml"]["rmse"], "r2": summary["best_tuning_piml"]["r2"]},
        "evaluation_Ea_eV": {
            "min": float(best_eval["predicted_Ea"].min()),
            "median": float(best_eval["predicted_Ea"].median()),
            "max": float(best_eval["predicted_Ea"].max()),
        },
        "note": "Candidate predictions are diagnostic only and do not replace archived-paper-checkpoint evidence.",
    }
    (run_dir / "optimized_physics_diagnostics.json").write_text(
        json.dumps(physics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(json.dumps(physics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
