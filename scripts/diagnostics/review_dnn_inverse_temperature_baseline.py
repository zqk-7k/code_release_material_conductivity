"""Lightweight DNN diagnostic with an inverse-temperature feature.

This script avoids the PyTorch training stack used for the locked manuscript
DNN because torch is not available in the current Windows environment. It uses
the same processed tabular fields reconstructed from the relational TSV files,
adds both temperature and 1000/T features, and trains a fixed scikit-learn MLP
without hyperparameter search. The result is a fairness diagnostic only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
DATA = ROOT / "windows_material_conductivity_training_handoff" / "training_project" / "material-conductivity-data-clean_reference" / "data"
OUT = PAPER / "results" / "review_validation_full"
TARGET = "log_conductivity"
TEMP = "temperature_kelvin"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def flatten_text(values):
    return values.squeeze()


def load_processed_frame():
    samples = pd.read_csv(DATA / "material_samples.tsv", sep="\t")
    dopants = pd.read_csv(DATA / "sample_dopants.tsv", sep="\t")
    sintering = pd.read_csv(DATA / "sintering_steps.tsv", sep="\t")

    dopants = dopants.rename(
        columns={
            "dopant_molar_fraction": "dopant_fraction",
            "dopant_ionic_radius": "ionic_radius",
        }
    )
    sintering = sintering.rename(
        columns={
            "sintering_temperature": "temperature_celsius",
            "sintering_duration": "duration_hours",
        }
    )

    for col in ["dopant_fraction", "ionic_radius", "dopant_valence"]:
        dopants[col] = pd.to_numeric(dopants[col], errors="coerce")
    for col in ["temperature_celsius", "duration_hours"]:
        sintering[col] = pd.to_numeric(sintering[col], errors="coerce")
    for col in ["operating_temperature", "conductivity"]:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")

    dparts = []
    for sid, g in dopants.groupby("sample_id"):
        fractions = g["dopant_fraction"].fillna(0).to_numpy(float)
        total_fraction = float(np.nansum(fractions))
        if total_fraction > 0:
            avg_radius = float(np.nansum(g["ionic_radius"].to_numpy(float) * fractions) / total_fraction)
            avg_valence = float(np.nansum(g["dopant_valence"].to_numpy(float) * fractions) / total_fraction)
            primary = str(g.iloc[int(np.nanargmax(fractions))]["dopant_element"])
        else:
            avg_radius = np.nan
            avg_valence = np.nan
            primary = str(g.iloc[0]["dopant_element"]) if len(g) else "missing"
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

    df = samples.merge(dopant_summary, on="sample_id", how="left").merge(sintering_summary, on="sample_id", how="left")
    df = df[(df["conductivity"] > 0) & df["operating_temperature"].notna()].copy()
    df[TEMP] = df["operating_temperature"] + 273.15
    df[TARGET] = np.log10(df["conductivity"])
    df["inv_temperature_1000_over_K"] = 1000.0 / df[TEMP]
    df["number_of_dopants"] = df["number_of_dopants"].fillna(0)
    return df.reset_index(drop=True)


def group_series(df):
    group_cols = [
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
    normalized = df[group_cols].copy()
    for col in normalized.select_dtypes(include=["float", "float32", "float64"]).columns:
        normalized[col] = normalized[col].round(6)
    return pd.util.hash_pandas_object(normalized.fillna("__MISSING__").astype(str), index=False).astype(str)


def feature_pipeline():
    numeric = [
        "total_dopant_fraction",
        "average_dopant_radius",
        "average_dopant_valence",
        "number_of_dopants",
        "maximum_sintering_temperature",
        "total_sintering_duration",
        TEMP,
        "inv_temperature_1000_over_K",
    ]
    categorical = ["synthesis_method", "primary_dopant_element"]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", onehot())]),
                categorical,
            ),
            (
                "text",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", FunctionTransformer(flatten_text, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
                        ("svd", TruncatedSVD(n_components=16, random_state=42)),
                    ]
                ),
                ["material_source_and_purity"],
            ),
        ]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_processed_frame()
    temperature_c = df[TEMP] - 273.15
    train = df[temperature_c <= 800.0 + 1e-6].copy()
    high = df[temperature_c >= 900.0 - 1e-6].copy()
    low_groups = set(group_series(train).tolist())
    high_groups = group_series(high)
    matched = high[high_groups.isin(low_groups)].copy()
    tests = {"all_high_temperature": high, "matched_material_high_temperature": matched}

    pipeline = feature_pipeline()
    x_train = pipeline.fit_transform(train)
    y_train = train[TARGET].to_numpy(float)

    seeds = [0, 1, 2, 42, 2026]
    predictions = {name: [] for name in tests}
    for seed in seeds:
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=7e-4,
            max_iter=190,
            random_state=seed,
            early_stopping=False,
            shuffle=True,
        )
        model.fit(x_train, y_train)
        for name, frame in tests.items():
            predictions[name].append(model.predict(pipeline.transform(frame)))

    rows = []
    for name, frame in tests.items():
        pred = np.mean(predictions[name], axis=0)
        y = frame[TARGET].to_numpy(float)
        rows.append(
            {
                "model": "sklearn_MLP_DNN_plus_temperature_and_1000_over_T",
                "subset": name,
                "n_samples": len(frame),
                "rmse": rmse(y, pred),
                "r2": float(r2_score(y, pred)),
                "protocol_note": "Lightweight fixed five-seed sklearn MLP diagnostic; same reconstructed processed descriptors plus temperature and 1000/T; no hyperparameter search; not a replacement for the locked PyTorch DNN.",
            }
        )
        out_pred = frame[["sample_id", TEMP, TARGET]].copy()
        out_pred["prediction"] = pred
        out_pred.to_csv(OUT / f"dnn_inverse_temperature_{name}_predictions.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "dnn_inverse_temperature_baseline.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
