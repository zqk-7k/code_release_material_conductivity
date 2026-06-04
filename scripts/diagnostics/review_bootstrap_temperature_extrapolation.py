from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
SRC = ROOT / "windows_material_conductivity_training_handoff" / "windows_experiments" / "piml_metric_optimization" / "results" / "review_validation_full" / "temperature_extrapolation"
DATA = ROOT / "windows_material_conductivity_training_handoff" / "training_project" / "material-conductivity-data-clean_reference" / "data"
OUT = PAPER / "results" / "review_validation_full"
FIG = PAPER / "figures"


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _fmt(value):
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).strip().lower()


def load_material_fingerprints():
    samples = pd.read_csv(DATA / "material_samples.tsv", sep="\t")
    dopants = pd.read_csv(DATA / "sample_dopants.tsv", sep="\t")
    sintering = pd.read_csv(DATA / "sintering_steps.tsv", sep="\t")

    rename_map = {
        "dopant_molar_fraction": "dopant_fraction",
        "dopant_ionic_radius": "ionic_radius",
        "sintering_temperature": "temperature_celsius",
        "sintering_duration": "duration_hours",
    }
    dopants = dopants.rename(columns={k: v for k, v in rename_map.items() if k in dopants.columns})
    sintering = sintering.rename(columns={k: v for k, v in rename_map.items() if k in sintering.columns})

    for col in ["dopant_fraction", "ionic_radius", "dopant_valence"]:
        if col in dopants:
            dopants[col] = dopants[col].map(_safe_float)
    if "temperature_celsius" in sintering:
        sintering["temperature_celsius"] = sintering["temperature_celsius"].map(_safe_float)
    if "duration_hours" in sintering:
        sintering["duration_hours"] = sintering["duration_hours"].map(_safe_float)

    dparts = []
    for sid, g in dopants.groupby("sample_id"):
        fractions = g.get("dopant_fraction", pd.Series(dtype=float)).fillna(0).to_numpy(float)
        total_fraction = float(np.nansum(fractions))
        if total_fraction > 0:
            radii = g.get("ionic_radius", pd.Series(dtype=float)).to_numpy(float)
            valences = g.get("dopant_valence", pd.Series(dtype=float)).to_numpy(float)
            avg_radius = float(np.nansum(radii * fractions) / total_fraction)
            avg_valence = float(np.nansum(valences * fractions) / total_fraction)
        else:
            avg_radius = np.nan
            avg_valence = np.nan

        primary = "unknown"
        if len(g) and "dopant_element" in g:
            if total_fraction > 0:
                primary = str(g.iloc[int(np.nanargmax(fractions))]["dopant_element"])
            else:
                primary = str(g.iloc[0]["dopant_element"])

        dparts.append(
            {
                "sample_id": sid,
                "total_dopant_fraction": total_fraction,
                "average_dopant_radius": avg_radius,
                "average_dopant_valence": avg_valence,
                "number_of_dopants": int(len(g)),
                "primary_dopant_element": primary,
            }
        )
    dopant_summary = pd.DataFrame(dparts)

    sintering_summary = (
        sintering.groupby("sample_id", as_index=False)
        .agg(
            maximum_sintering_temperature=("temperature_celsius", "max"),
            total_sintering_duration=("duration_hours", "sum"),
        )
    )

    merged = samples.merge(dopant_summary, on="sample_id", how="left").merge(sintering_summary, on="sample_id", how="left")
    fields = [
        "material_source_and_purity",
        "synthesis_method",
        "total_dopant_fraction",
        "average_dopant_radius",
        "average_dopant_valence",
        "number_of_dopants",
        "primary_dopant_element",
        "maximum_sintering_temperature",
        "total_sintering_duration",
    ]
    merged["material_fingerprint"] = merged[fields].apply(lambda row: "|".join(_fmt(row.get(f)) for f in fields), axis=1)
    return merged[["sample_id", "material_fingerprint"]]


def bootstrap_by_record(y, p_piml, p_dnn, n_boot=10000, seed=20260529):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = rmse(y[idx], p_dnn[idx]) - rmse(y[idx], p_piml[idx])
    return deltas


def bootstrap_by_fingerprint(df, y, p_piml, p_dnn, n_boot=10000, seed=20260530):
    rng = np.random.default_rng(seed)
    group_to_indices = {
        fp: np.flatnonzero(df["material_fingerprint"].to_numpy() == fp)
        for fp in df["material_fingerprint"].drop_duplicates()
    }
    groups = np.array(list(group_to_indices.keys()), dtype=object)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        deltas[i] = rmse(y[idx], p_dnn[idx]) - rmse(y[idx], p_piml[idx])
    return deltas, len(groups)


def summarize_bootstrap(unit, deltas, n, n_fingerprints, observed_piml, observed_dnn, observed_delta, n_boot):
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    return {
        "subset": "matched_material_high_temperature",
        "bootstrap_unit": unit,
        "n_samples": n,
        "n_material_fingerprints": n_fingerprints,
        "n_bootstrap": n_boot,
        "piml_rmse": observed_piml,
        "dnn_rmse": observed_dnn,
        "delta_rmse_dnn_minus_piml": observed_delta,
        "delta_rmse_ci95_low": float(ci_low),
        "delta_rmse_ci95_high": float(ci_high),
        "bootstrap_probability_delta_le_0": float(np.mean(deltas <= 0)),
        "source_piml_predictions": str(SRC / "piml_full_matched_material_high_temperature_predictions.csv"),
        "source_dnn_predictions": str(SRC / "dnn_full_matched_material_high_temperature_predictions.csv"),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    piml = pd.read_csv(SRC / "piml_full_matched_material_high_temperature_predictions.csv")
    dnn = pd.read_csv(SRC / "dnn_full_matched_material_high_temperature_predictions.csv")

    keep = ["sample_id", "temperature_kelvin", "log_conductivity", "prediction"]
    merged = piml[keep].merge(
        dnn[keep],
        on=["sample_id", "temperature_kelvin", "log_conductivity"],
        suffixes=("_piml", "_dnn"),
        validate="one_to_one",
    )
    if len(merged) != 47:
        raise RuntimeError(f"Expected matched-material n=47, found n={len(merged)}")

    merged = merged.merge(load_material_fingerprints(), on="sample_id", how="left")
    if merged["material_fingerprint"].isna().any():
        missing = merged.loc[merged["material_fingerprint"].isna(), "sample_id"].tolist()
        raise RuntimeError(f"Missing fingerprints for sample_id values: {missing}")

    y = merged["log_conductivity"].to_numpy(float)
    p_piml = merged["prediction_piml"].to_numpy(float)
    p_dnn = merged["prediction_dnn"].to_numpy(float)

    observed_piml = rmse(y, p_piml)
    observed_dnn = rmse(y, p_dnn)
    observed_delta = observed_dnn - observed_piml
    n_boot = 10000

    record_deltas = bootstrap_by_record(y, p_piml, p_dnn, n_boot=n_boot)
    fingerprint_deltas, n_fingerprints = bootstrap_by_fingerprint(merged, y, p_piml, p_dnn, n_boot=n_boot)

    summary = pd.DataFrame(
        [
            summarize_bootstrap("record_paired", record_deltas, len(merged), n_fingerprints, observed_piml, observed_dnn, observed_delta, n_boot),
            summarize_bootstrap(
                "material_fingerprint_paired",
                fingerprint_deltas,
                len(merged),
                n_fingerprints,
                observed_piml,
                observed_dnn,
                observed_delta,
                n_boot,
            ),
        ]
    )
    summary.to_csv(OUT / "temperature_extrapolation_bootstrap_ci.csv", index=False)
    summary.to_csv(OUT / "temperature_extrapolation_grouped_bootstrap_ci.csv", index=False)
    merged.to_csv(OUT / "temperature_extrapolation_bootstrap_matched_predictions.csv", index=False)

    influence = merged.copy()
    influence["piml_sqerr"] = (influence["log_conductivity"] - influence["prediction_piml"]) ** 2
    influence["dnn_sqerr"] = (influence["log_conductivity"] - influence["prediction_dnn"]) ** 2
    influence["delta_sqerr_dnn_minus_piml"] = influence["dnn_sqerr"] - influence["piml_sqerr"]
    influence["abs_delta_sqerr"] = influence["delta_sqerr_dnn_minus_piml"].abs()
    influence = influence.sort_values("abs_delta_sqerr", ascending=False)
    influence.to_csv(OUT / "temperature_extrapolation_record_influence.csv", index=False)

    influence_rows = []
    ordered_idx = influence.index.to_list()
    for k in [0, 1, 2, 3]:
        kept = merged.drop(index=ordered_idx[:k])
        kept_piml = rmse(kept["log_conductivity"], kept["prediction_piml"])
        kept_dnn = rmse(kept["log_conductivity"], kept["prediction_dnn"])
        influence_rows.append(
            {
                "removed_top_abs_influence_records": k,
                "n_samples": len(kept),
                "piml_rmse": kept_piml,
                "dnn_rmse": kept_dnn,
                "delta_rmse_dnn_minus_piml": kept_dnn - kept_piml,
            }
        )
    pd.DataFrame(influence_rows).to_csv(OUT / "temperature_extrapolation_outlier_influence.csv", index=False)

    plt.figure(figsize=(5.2, 3.5))
    plt.hist(record_deltas, bins=50, color="#4C78A8", alpha=0.85, edgecolor="white")
    plt.axvline(observed_delta, color="#D62728", lw=2, label=f"Observed delta RMSE = {observed_delta:.3f}")
    plt.axvline(0, color="black", lw=1, ls="--")
    ci_low, ci_high = np.quantile(record_deltas, [0.025, 0.975])
    plt.axvspan(ci_low, ci_high, color="#4C78A8", alpha=0.15, label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    plt.xlabel("delta RMSE = RMSE(DNN) - RMSE(PIML)")
    plt.ylabel("Bootstrap count")
    plt.title("Matched-material temperature extrapolation")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "temperature_extrapolation_bootstrap_delta_rmse.pdf")
    plt.savefig(FIG / "temperature_extrapolation_bootstrap_delta_rmse.png", dpi=300)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
