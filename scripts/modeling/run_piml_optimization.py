"""Isolated optimization workflow for strict Arrhenius-constrained PIML models.

Two evaluation protocols are intentionally separated:

* ``paper_compatible`` reproduces the historical row-wise 80/20 split so its
  scores can be compared with the current manuscript Table 1.  It is useful
  for diagnosis, but it is not an untouched-test estimate after tuning.
* ``strict_grouped`` groups repeated material/process records, tunes only on
  an inner validation partition, and evaluates the selected models once on an
  outer test partition.

The script never writes into packaged ``training_project/results`` or archived
checkpoint directories.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PACKAGE_ROOT / "training_project" / "material-conductivity-data-analysis-ml"
SRC_DIR = ML_ROOT / "src" / "zirconia"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from etl.material_data_processor import MaterialDataProcessor  # noqa: E402
from models.baseline_net import StandardDNN  # noqa: E402
from models.piml_net import KB_EV, PhysicsInformedNet  # noqa: E402

PAPER_DNN_RMSE = 0.2518339612053054
PAPER_DNN_R2 = 0.9360484768769898
SEED = 42
TARGET = "log_conductivity"
TEMP = "temperature_kelvin"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    widths: tuple[int, ...] = (128, 64, 32)
    activation: str = "relu"
    dropout: float = 0.2
    ea_mode: str = "softplus"
    ea_min: float = 0.03
    ea_max: float = 2.0
    ea_init: float | None = None
    loga_init: float | None = None
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    loss: str = "mse"
    scheduler: bool = False
    legacy_model: bool = False


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_text_column(values: np.ndarray) -> np.ndarray:
    return values.squeeze()


flatten_text_column.__module__ = "run_piml_optimization"
sys.modules.setdefault("run_piml_optimization", sys.modules[__name__])


def build_serializable_feature_pipeline() -> ColumnTransformer:
    numeric = [
        "total_dopant_fraction",
        "average_dopant_radius",
        "average_dopant_valence",
        "number_of_dopants",
        "maximum_sintering_temperature",
        "total_sintering_duration",
    ]
    categorical = ["synthesis_method", "primary_dopant_element"]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "text",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", FunctionTransformer(flatten_text_column, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
                        ("svd", TruncatedSVD(n_components=16, random_state=SEED)),
                    ]
                ),
                ["material_source_and_purity"],
            ),
        ]
    )


def activation_layer(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU()}[name]


class TunablePIML(nn.Module):
    """Arrhenius-constrained network with configurable stable parameter heads."""

    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_dim
        for idx, width in enumerate(cfg.widths):
            blocks.extend([nn.Linear(current, width), nn.BatchNorm1d(width), activation_layer(cfg.activation)])
            if cfg.dropout and idx < len(cfg.widths) - 1:
                blocks.append(nn.Dropout(cfg.dropout))
            current = width
        self.encoder = nn.Sequential(*blocks)
        self.ea_head = nn.Linear(current, 1)
        self.loga_head = nn.Linear(current, 1)
        self.cfg = cfg
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        if self.cfg.ea_init is not None:
            if self.cfg.ea_mode == "bounded":
                fraction = (self.cfg.ea_init - self.cfg.ea_min) / (self.cfg.ea_max - self.cfg.ea_min)
                fraction = min(max(fraction, 1e-4), 1 - 1e-4)
                bias = math.log(fraction / (1 - fraction))
            else:
                bias = math.log(math.expm1(self.cfg.ea_init))
            nn.init.constant_(self.ea_head.bias, bias)
        if self.cfg.loga_init is not None:
            nn.init.constant_(self.loga_head.bias, self.cfg.loga_init)

    def forward(self, features: torch.Tensor, temperature_k: torch.Tensor):
        hidden = self.encoder(features)
        raw_ea = self.ea_head(hidden)
        if self.cfg.ea_mode == "bounded":
            ea = self.cfg.ea_min + (self.cfg.ea_max - self.cfg.ea_min) * torch.sigmoid(raw_ea)
        else:
            ea = torch.nn.functional.softplus(raw_ea)
        loga = self.loga_head(hidden)
        prediction = loga - torch.log10(temperature_k) - ea / (KB_EV * temperature_k * math.log(10))
        return prediction, ea, loga


def create_piml(input_dim: int, cfg: ModelConfig) -> nn.Module:
    return PhysicsInformedNet(input_dim) if cfg.legacy_model else TunablePIML(input_dim, cfg)


def model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(name="legacy_exact", legacy_model=True),
        ModelConfig(name="relu_softplus_init", ea_init=0.75, loga_init=5.0, dropout=0.10, lr=8e-4),
        ModelConfig(name="silu_softplus_init", activation="silu", ea_init=0.75, loga_init=5.0, dropout=0.10, lr=8e-4),
        ModelConfig(name="gelu_softplus_init", activation="gelu", ea_init=0.75, loga_init=5.0, dropout=0.10, lr=8e-4),
        ModelConfig(name="silu_bounded", activation="silu", ea_mode="bounded", ea_init=0.75, loga_init=5.0, dropout=0.10, lr=8e-4),
        ModelConfig(name="relu_bounded", ea_mode="bounded", ea_init=0.75, loga_init=5.0, dropout=0.05, lr=8e-4),
        ModelConfig(name="wide_silu_bounded", widths=(256, 128, 64), activation="silu", ea_mode="bounded", ea_init=0.75, loga_init=5.0, dropout=0.10, lr=6e-4),
        ModelConfig(name="wide_relu_softplus", widths=(256, 128, 64), ea_init=0.75, loga_init=5.0, dropout=0.10, lr=6e-4),
        ModelConfig(name="compact_silu_bounded", widths=(96, 48, 24), activation="silu", ea_mode="bounded", ea_init=0.75, loga_init=5.0, dropout=0.05, lr=1e-3),
        ModelConfig(name="silu_bounded_huber", activation="silu", ea_mode="bounded", ea_init=0.75, loga_init=5.0, dropout=0.10, lr=8e-4, loss="huber"),
        ModelConfig(name="silu_bounded_lowwd", activation="silu", ea_mode="bounded", ea_init=0.65, loga_init=4.5, dropout=0.05, lr=1e-3, weight_decay=1e-5, scheduler=True),
        ModelConfig(name="wide_gelu_bounded", widths=(256, 128, 32), activation="gelu", ea_mode="bounded", ea_init=0.65, loga_init=4.5, dropout=0.05, lr=8e-4, weight_decay=1e-5, scheduler=True),
    ]


def build_group_fingerprint(df: pd.DataFrame) -> pd.Series:
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


def prepare_split(
    df: pd.DataFrame, protocol: str, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if protocol == "paper_compatible":
        train, evaluation = train_test_split(df, test_size=0.2, random_state=split_seed)
        return train.copy(), evaluation.copy(), None
    groups = build_group_fingerprint(df)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed)
    development_idx, test_idx = next(outer.split(df, groups=groups))
    development = df.iloc[development_idx].copy()
    test = df.iloc[test_idx].copy()
    dev_groups = groups.iloc[development_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed)
    train_idx, validation_idx = next(inner.split(development, groups=dev_groups))
    return development.iloc[train_idx].copy(), development.iloc[validation_idx].copy(), test


def transform_frames(train: pd.DataFrame, evaluation: pd.DataFrame):
    pipeline = build_serializable_feature_pipeline()
    set_seed(SEED)
    x_train = pipeline.fit_transform(train)
    x_eval = pipeline.transform(evaluation)
    return pipeline, x_train, x_eval


def tensors(x: np.ndarray, frame: pd.DataFrame) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(frame[TEMP].to_numpy(), dtype=torch.float32).view(-1, 1),
        torch.as_tensor(frame[TARGET].to_numpy(), dtype=torch.float32).view(-1, 1),
    )


def predict_piml(model: nn.Module, x: np.ndarray, frame: pd.DataFrame, device: torch.device):
    model.eval()
    with torch.no_grad():
        pred, ea, loga = model(
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(frame[TEMP].to_numpy(), dtype=torch.float32, device=device).view(-1, 1),
        )
    return pred.cpu().numpy().ravel(), ea.cpu().numpy().ravel(), loga.cpu().numpy().ravel()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def train_piml_trial(
    cfg: ModelConfig,
    seed: int,
    x_train: np.ndarray,
    train: pd.DataFrame,
    x_eval: np.ndarray,
    evaluation: pd.DataFrame,
    out_dir: Path,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    set_seed(seed)
    model = create_piml(x_train.shape[1], cfg).to(device)
    criterion: nn.Module = nn.MSELoss() if cfg.loss == "mse" else nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, min_lr=1e-5)
        if cfg.scheduler
        else None
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tensors(x_train, train), batch_size=cfg.batch_size, shuffle=True, generator=generator)
    y_eval = evaluation[TARGET].to_numpy()
    best_mse = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    stagnant = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for features, temperature, target in loader:
            features, temperature, target = features.to(device), temperature.to(device), target.to(device)
            optimizer.zero_grad()
            prediction, _, _ = model(features, temperature)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        prediction, _, _ = predict_piml(model, x_eval, evaluation, device)
        eval_mse = float(mean_squared_error(y_eval, prediction))
        if scheduler is not None:
            scheduler.step(eval_mse)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "eval_mse": eval_mse})
        if eval_mse < best_mse - 1e-8:
            best_mse = eval_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stagnant = 0
        else:
            stagnant += 1
        if epoch % 50 == 0:
            print(
                f"  {cfg.name} seed={seed} epoch={epoch} eval_rmse={math.sqrt(eval_mse):.6f} "
                f"best_rmse={math.sqrt(best_mse):.6f}",
                flush=True,
            )
        if stagnant >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for {cfg.name} seed {seed}")
    model.load_state_dict(best_state)
    prediction, ea, loga = predict_piml(model, x_eval, evaluation, device)
    trial_dir = out_dir / "checkpoints" / f"{cfg.name}_seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = trial_dir / "best_piml_model.pth"
    torch.save({"state_dict": best_state, "config": asdict(cfg), "input_dim": x_train.shape[1]}, checkpoint)
    pd.DataFrame(history).to_csv(trial_dir / "training_history.csv", index=False)
    prediction_frame = evaluation[["sample_id", TEMP, TARGET]].copy()
    prediction_frame["prediction"] = prediction
    prediction_frame["predicted_Ea"] = ea
    prediction_frame["predicted_logA"] = loga
    prediction_frame.to_csv(trial_dir / "evaluation_predictions.csv", index=False)
    score = metrics(y_eval, prediction)
    result = {
        "model": cfg.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        **score,
        "beats_paper_dnn_rmse": score["rmse"] < PAPER_DNN_RMSE,
        "beats_paper_dnn_r2": score["r2"] > PAPER_DNN_R2,
        "checkpoint": str(checkpoint.relative_to(out_dir)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_json": json.dumps(asdict(cfg), ensure_ascii=True),
    }
    return result, prediction


def train_dnn_reference(
    seed: int,
    x_train: np.ndarray,
    train: pd.DataFrame,
    x_eval: np.ndarray,
    evaluation: pd.DataFrame,
    out_dir: Path,
    epochs: int,
    patience: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    scaler = StandardScaler()
    t_train = scaler.fit_transform(train[[TEMP]])
    t_eval = scaler.transform(evaluation[[TEMP]])
    model = StandardDNN(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(t_train, dtype=torch.float32),
            torch.as_tensor(train[TARGET].to_numpy(), dtype=torch.float32).view(-1, 1),
        ),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    y_eval = evaluation[TARGET].to_numpy()
    best_mse = float("inf")
    best_state = None
    best_epoch = -1
    stagnant = 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for features, temperature, target in loader:
            features, temperature, target = features.to(device), temperature.to(device), target.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(features, temperature), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.as_tensor(x_eval, dtype=torch.float32, device=device),
                torch.as_tensor(t_eval, dtype=torch.float32, device=device),
            ).cpu().numpy().ravel()
        eval_mse = float(mean_squared_error(y_eval, pred))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "eval_mse": eval_mse})
        if eval_mse < best_mse - 1e-8:
            best_mse, best_epoch, best_state, stagnant = eval_mse, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stagnant += 1
        if epoch % 50 == 0:
            print(
                f"  StandardDNN_reference seed={seed} epoch={epoch} eval_rmse={math.sqrt(eval_mse):.6f} "
                f"best_rmse={math.sqrt(best_mse):.6f}",
                flush=True,
            )
        if stagnant >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.as_tensor(x_eval, dtype=torch.float32, device=device),
            torch.as_tensor(t_eval, dtype=torch.float32, device=device),
        ).cpu().numpy().ravel()
    reference_dir = out_dir / "checkpoints" / f"dnn_reference_seed_{seed}"
    reference_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reference_dir / "best_dnn_model.pth"
    torch.save({"state_dict": best_state, "input_dim": x_train.shape[1]}, ckpt)
    pd.DataFrame(history).to_csv(reference_dir / "training_history.csv", index=False)
    frame = evaluation[["sample_id", TEMP, TARGET]].copy()
    frame["prediction"] = pred
    frame.to_csv(reference_dir / "evaluation_predictions.csv", index=False)
    return {
        "model": "StandardDNN_reference",
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        **metrics(y_eval, pred),
        "checkpoint": str(ckpt.relative_to(out_dir)),
        "checkpoint_sha256": sha256_file(ckpt),
    }


def train_final_piml(
    cfg: ModelConfig,
    seed: int,
    epochs: int,
    x_development: np.ndarray,
    development: pd.DataFrame,
    x_test: np.ndarray,
    test: pd.DataFrame,
    out_dir: Path,
    tag: str,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    set_seed(seed)
    model = create_piml(x_development.shape[1], cfg).to(device)
    criterion: nn.Module = nn.MSELoss() if cfg.loss == "mse" else nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        tensors(x_development, development),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for features, temperature, target in loader:
            features, temperature, target = features.to(device), temperature.to(device), target.to(device)
            optimizer.zero_grad()
            prediction, _, _ = model(features, temperature)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    prediction, ea, loga = predict_piml(model, x_test, test, device)
    final_dir = out_dir / "final_outer_test" / tag
    final_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = final_dir / "piml_model.pth"
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "input_dim": x_development.shape[1]}, checkpoint)
    frame = test[["sample_id", TEMP, TARGET]].copy()
    frame["prediction"] = prediction
    frame["predicted_Ea"] = ea
    frame["predicted_logA"] = loga
    frame.to_csv(final_dir / "test_predictions.csv", index=False)
    result = {
        "model": cfg.name,
        "seed": seed,
        "fixed_epochs_from_inner_validation": epochs,
        **metrics(test[TARGET].to_numpy(), prediction),
        "checkpoint": str(checkpoint.relative_to(out_dir)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return result, prediction


def train_final_dnn(
    seed: int,
    epochs: int,
    x_development: np.ndarray,
    development: pd.DataFrame,
    x_test: np.ndarray,
    test: pd.DataFrame,
    out_dir: Path,
    device: torch.device,
) -> dict:
    set_seed(seed)
    scaler = StandardScaler()
    t_dev = scaler.fit_transform(development[[TEMP]])
    t_test = scaler.transform(test[[TEMP]])
    model = StandardDNN(x_development.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_development, dtype=torch.float32),
            torch.as_tensor(t_dev, dtype=torch.float32),
            torch.as_tensor(development[TARGET].to_numpy(), dtype=torch.float32).view(-1, 1),
        ),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for features, temperature, target in loader:
            features, temperature, target = features.to(device), temperature.to(device), target.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(features, temperature), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.as_tensor(x_test, dtype=torch.float32, device=device),
            torch.as_tensor(t_test, dtype=torch.float32, device=device),
        ).cpu().numpy().ravel()
    final_dir = out_dir / "final_outer_test" / "dnn_reference"
    final_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = final_dir / "dnn_model.pth"
    torch.save({"state_dict": model.state_dict(), "input_dim": x_development.shape[1]}, checkpoint)
    frame = test[["sample_id", TEMP, TARGET]].copy()
    frame["prediction"] = prediction
    frame.to_csv(final_dir / "test_predictions.csv", index=False)
    return {
        "model": "StandardDNN_reference",
        "seed": seed,
        "fixed_epochs_from_inner_validation": epochs,
        **metrics(test[TARGET].to_numpy(), prediction),
        "checkpoint": str(checkpoint.relative_to(out_dir)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def greedy_ensemble(
    trial_rows: list[dict],
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
    maximum: int = 5,
) -> tuple[list[str], np.ndarray, dict[str, float]]:
    ordered = [f"{row['model']}__seed_{row['seed']}" for row in sorted(trial_rows, key=lambda x: x["rmse"])]
    selected: list[str] = []
    best_pred = None
    best_score = float("inf")
    remaining = ordered.copy()
    while remaining and len(selected) < maximum:
        candidate_best = None
        candidate_pred = None
        candidate_score = best_score
        for key in remaining:
            keys = selected + [key]
            pred = np.mean([predictions[k] for k in keys], axis=0)
            score = float(np.sqrt(mean_squared_error(y, pred)))
            if score < candidate_score - 1e-9:
                candidate_best, candidate_pred, candidate_score = key, pred, score
        if candidate_best is None:
            break
        selected.append(candidate_best)
        remaining.remove(candidate_best)
        best_pred, best_score = candidate_pred, candidate_score
    if best_pred is None:
        selected = [ordered[0]]
        best_pred = predictions[selected[0]]
    return selected, best_pred, metrics(y, best_pred)


def run_protocol(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    df = MaterialDataProcessor().load_and_preprocess_data_for_training_piml()
    train, evaluation, outer_test = prepare_split(df, args.protocol, args.split_seed)
    output = Path(args.output) if args.output else Path(__file__).parent / "results" / f"{args.protocol}_{timestamp()}"
    output.mkdir(parents=True, exist_ok=True)
    x_pipeline, x_train, x_eval = transform_frames(train, evaluation)
    joblib.dump(x_pipeline, output / "tuning_preprocessor.joblib")
    split_rows = []
    for split_name, frame in [("train", train), ("evaluation", evaluation), ("outer_test", outer_test)]:
        if frame is not None:
            rows = frame[["sample_id"]].copy()
            rows["split"] = split_name
            split_rows.append(rows)
    pd.concat(split_rows).to_csv(output / "split_membership.csv", index=False)
    configs = model_configs()
    if args.configs:
        requested = {value.strip() for value in args.configs.split(",") if value.strip()}
        configs = [cfg for cfg in configs if cfg.name in requested]
        missing = requested - {cfg.name for cfg in configs}
        if missing:
            raise ValueError(f"Unknown configs: {sorted(missing)}")
    if args.max_configs:
        configs = configs[: args.max_configs]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    trial_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for cfg in configs:
        for seed in seeds:
            result, pred = train_piml_trial(
                cfg, seed, x_train, train, x_eval, evaluation, output, args.epochs, args.patience, device
            )
            trial_rows.append(result)
            predictions[f"{cfg.name}__seed_{seed}"] = pred
            pd.DataFrame(trial_rows).sort_values("rmse").to_csv(output / "piml_trials.csv", index=False)
            print(
                f"[PIML] {cfg.name} seed={seed} RMSE={result['rmse']:.6f} "
                f"R2={result['r2']:.6f} best_epoch={result['best_epoch']}",
                flush=True,
            )
    y_eval = evaluation[TARGET].to_numpy()
    selected, ensemble_pred, ensemble_score = greedy_ensemble(trial_rows, predictions, y_eval)
    ensemble_frame = evaluation[["sample_id", TEMP, TARGET]].copy()
    ensemble_frame["prediction"] = ensemble_pred
    ensemble_frame.to_csv(output / "piml_ensemble_predictions.csv", index=False)
    dnn_reference = train_dnn_reference(
        args.dnn_seed, x_train, train, x_eval, evaluation, output, args.epochs, args.patience, device
    )
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": args.protocol,
        "split_seed": args.split_seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "n_rows": len(df),
        "n_train": len(train),
        "n_evaluation": len(evaluation),
        "n_outer_test_reserved": 0 if outer_test is None else len(outer_test),
        "paper_dnn_threshold": {"rmse": PAPER_DNN_RMSE, "r2": PAPER_DNN_R2},
        "best_tuning_piml": min(trial_rows, key=lambda row: row["rmse"]),
        "tuning_ensemble": {
            "members": selected,
            **ensemble_score,
            "beats_paper_dnn_rmse": ensemble_score["rmse"] < PAPER_DNN_RMSE,
            "beats_paper_dnn_r2": ensemble_score["r2"] > PAPER_DNN_R2,
        },
        "same_split_dnn_reference": dnn_reference,
        "caution": (
            "Historical-table-compatible evaluation is reused for tuning; do not describe as independent test performance."
            if args.protocol == "paper_compatible"
            else "Outer test is reserved and not evaluated by this tuning command."
        ),
    }
    if args.protocol == "strict_grouped" and outer_test is not None:
        development = pd.concat([train, evaluation], axis=0).copy()
        final_pipeline, x_development, x_test = transform_frames(development, outer_test)
        joblib.dump(final_pipeline, output / "final_outer_test_preprocessor.joblib")
        best_tuning = min(trial_rows, key=lambda row: row["rmse"])
        config_map = {cfg.name: cfg for cfg in configs}
        best_final, _ = train_final_piml(
            config_map[best_tuning["model"]],
            int(best_tuning["seed"]),
            int(best_tuning["best_epoch"]) + 1,
            x_development,
            development,
            x_test,
            outer_test,
            output,
            "best_single",
            device,
        )
        final_member_results = []
        final_member_predictions = []
        for member in selected:
            model_name, seed_text = member.rsplit("__seed_", maxsplit=1)
            tuning_row = next(row for row in trial_rows if row["model"] == model_name and row["seed"] == int(seed_text))
            member_result, member_prediction = train_final_piml(
                config_map[model_name],
                int(seed_text),
                int(tuning_row["best_epoch"]) + 1,
                x_development,
                development,
                x_test,
                outer_test,
                output,
                f"ensemble_{model_name}_seed_{seed_text}",
                device,
            )
            final_member_results.append(member_result)
            final_member_predictions.append(member_prediction)
        outer_ensemble_prediction = np.mean(final_member_predictions, axis=0)
        outer_ensemble_score = metrics(outer_test[TARGET].to_numpy(), outer_ensemble_prediction)
        outer_frame = outer_test[["sample_id", TEMP, TARGET]].copy()
        outer_frame["prediction"] = outer_ensemble_prediction
        outer_frame.to_csv(output / "final_outer_test" / "piml_ensemble_test_predictions.csv", index=False)
        final_dnn = train_final_dnn(
            args.dnn_seed,
            int(dnn_reference["best_epoch"]) + 1,
            x_development,
            development,
            x_test,
            outer_test,
            output,
            device,
        )
        summary["final_outer_test"] = {
            "n_test": len(outer_test),
            "selected_without_test_access": True,
            "best_single_piml": best_final,
            "piml_ensemble_members": final_member_results,
            "piml_ensemble": outer_ensemble_score,
            "dnn_reference": final_dnn,
            "ensemble_beats_dnn_rmse": outer_ensemble_score["rmse"] < final_dnn["rmse"],
            "ensemble_beats_dnn_r2": outer_ensemble_score["r2"] > final_dnn["r2"],
            "single_beats_dnn_rmse": best_final["rmse"] < final_dnn["rmse"],
            "single_beats_dnn_r2": best_final["r2"] > final_dnn["r2"],
        }
    (output / "tuning_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["paper_compatible", "strict_grouped"], default="paper_compatible")
    parser.add_argument("--output")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--configs", default="")
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--dnn-seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_protocol(args)


if __name__ == "__main__":
    main()
