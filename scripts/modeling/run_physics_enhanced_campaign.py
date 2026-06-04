"""Nested grouped evaluation for physics-enhanced Arrhenius PIML versus tuned DNN.

The campaign reserves two confirmation partitions before model selection.
Configuration screening happens only inside the development partition via
grouped cross-validation. The selected PIML and DNN configurations are then
refit as fixed multi-seed ensembles and evaluated on the untouched partitions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_piml_optimization as base  # noqa: E402

KB_EV = 8.617333262e-5
TEMP = "temperature_kelvin"
TARGET = "log_conductivity"
CONFIG_SEED = 42
PARTITION_SEEDS = {"confirmation_b": 8101, "confirmation_a": 8102, "development_cv": 1907}
FINAL_SEEDS = [0, 1, 2, 42, 2026]


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    widths: tuple[int, ...] = (128, 64, 32)
    norm: str = "batch"
    activation: str = "relu"
    dropout: float = 0.10
    ea_mode: str = "softplus"
    ea_aux_weight: float = 0.0
    group_balanced: bool = False
    loss: str = "mse"
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    ea_init: float = 0.75
    loga_init: float = 5.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU()}[name]


def normalization(name: str, width: int) -> nn.Module:
    return {"batch": nn.BatchNorm1d(width), "layer": nn.LayerNorm(width), "none": nn.Identity()}[name]


class Encoder(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for idx, width in enumerate(cfg.widths):
            layers.extend([nn.Linear(current, width), normalization(cfg.norm, width), activation(cfg.activation)])
            if cfg.dropout > 0 and idx < len(cfg.widths) - 1:
                layers.append(nn.Dropout(cfg.dropout))
            current = width
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancedPIML(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(input_dim, cfg)
        hidden = cfg.widths[-1]
        self.ea_head = nn.Linear(hidden, 1)
        self.loga_head = nn.Linear(hidden, 1)
        if cfg.ea_mode == "bounded":
            fraction = (cfg.ea_init - 0.03) / (2.0 - 0.03)
            nn.init.constant_(self.ea_head.bias, math.log(fraction / (1 - fraction)))
        else:
            nn.init.constant_(self.ea_head.bias, math.log(math.expm1(cfg.ea_init)))
        nn.init.constant_(self.loga_head.bias, cfg.loga_init)

    def forward(self, x: torch.Tensor, temperature_k: torch.Tensor):
        hidden = self.encoder(x)
        raw_ea = self.ea_head(hidden)
        if self.cfg.ea_mode == "bounded":
            ea = 0.03 + (2.0 - 0.03) * torch.sigmoid(raw_ea)
        else:
            ea = nn.functional.softplus(raw_ea)
        loga = self.loga_head(hidden)
        prediction = loga - torch.log10(temperature_k) - ea / (KB_EV * temperature_k * math.log(10))
        return prediction, ea, loga


class TunedDNN(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.encoder = Encoder(input_dim, cfg)
        hidden = cfg.widths[-1]
        self.output = nn.Sequential(nn.Linear(hidden + 1, max(16, hidden // 2)), activation(cfg.activation), nn.Linear(max(16, hidden // 2), 1))

    def forward(self, x: torch.Tensor, temperature_scaled: torch.Tensor):
        return self.output(torch.cat([self.encoder(x), temperature_scaled], dim=1))


def configurations() -> list[Config]:
    piml = [
        Config("p_batch_softplus", "piml"),
        Config("p_batch_bounded", "piml", ea_mode="bounded", dropout=0.05),
        Config("p_batch_softplus_bal", "piml", group_balanced=True),
        Config("p_batch_bounded_bal", "piml", ea_mode="bounded", dropout=0.05, group_balanced=True),
        Config("p_layer_softplus_bal", "piml", norm="layer", group_balanced=True),
        Config("p_layer_bounded_bal", "piml", norm="layer", ea_mode="bounded", dropout=0.05, group_balanced=True),
        Config("p_layer_bounded_ea01", "piml", norm="layer", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.01),
        Config("p_layer_bounded_ea05", "piml", norm="layer", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.05),
        Config("p_batch_bounded_ea01", "piml", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.01),
        Config("p_batch_bounded_ea05", "piml", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.05),
        Config("p_layer_huber_ea01", "piml", norm="layer", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.01, loss="huber"),
        Config("p_wide_layer_ea01", "piml", widths=(256, 128, 64), norm="layer", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.01, lr=6e-4),
        Config("p_layer_silu_ea01", "piml", norm="layer", activation="silu", ea_mode="bounded", dropout=0.05, group_balanced=True, ea_aux_weight=0.01),
    ]
    dnn = [
        Config("d_batch_mse", "dnn", lr=1e-3),
        Config("d_batch_bal", "dnn", lr=1e-3, group_balanced=True),
        Config("d_layer_bal", "dnn", norm="layer", lr=1e-3, group_balanced=True),
        Config("d_layer_huber_bal", "dnn", norm="layer", lr=1e-3, group_balanced=True, loss="huber"),
        Config("d_wide_layer_bal", "dnn", widths=(256, 128, 64), norm="layer", lr=7e-4, group_balanced=True),
        Config("d_layer_silu_bal", "dnn", norm="layer", activation="silu", lr=1e-3, group_balanced=True),
        Config("d_none_bal", "dnn", norm="none", lr=1e-3, group_balanced=True),
    ]
    return piml + dnn


def group_series(df: pd.DataFrame) -> pd.Series:
    return base.build_group_fingerprint(df).set_axis(df.index)


def reserve_partitions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = group_series(df)
    split_b = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=PARTITION_SEEDS["confirmation_b"])
    remaining_idx, b_idx = next(split_b.split(df, groups=groups))
    remaining = df.iloc[remaining_idx].copy()
    b = df.iloc[b_idx].copy()
    groups_remaining = groups.iloc[remaining_idx]
    split_a = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=PARTITION_SEEDS["confirmation_a"])
    dev_idx, a_idx = next(split_a.split(remaining, groups=groups_remaining))
    return {"development": remaining.iloc[dev_idx].copy(), "confirmation_a": remaining.iloc[a_idx].copy(), "confirmation_b": b}


def empirical_ea_targets(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    groups = group_series(train)
    targets = np.full(len(train), np.nan, dtype=np.float32)
    group_values = []
    for _, positions in train.groupby(groups).groups.items():
        frame = train.loc[positions]
        if frame[TEMP].nunique() < 2:
            continue
        x = 1.0 / frame[TEMP].to_numpy(dtype=float)
        y = frame[TARGET].to_numpy(dtype=float) + np.log10(frame[TEMP].to_numpy(dtype=float))
        ea = float(-np.polyfit(x, y, 1)[0] * KB_EV * math.log(10))
        if 0.03 <= ea <= 2.0:
            local_indices = train.index.get_indexer(positions)
            targets[local_indices] = ea
            group_values.append(ea)
    mask = ~np.isnan(targets)
    details = {
        "n_rows_with_empirical_ea": int(mask.sum()),
        "n_groups_with_empirical_ea": len(group_values),
        "median_empirical_ea": float(np.median(group_values)) if group_values else None,
    }
    return np.nan_to_num(targets, nan=0.0), mask.astype(np.float32), details


def group_weights(train: pd.DataFrame, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones(len(train), dtype=np.float32)
    groups = group_series(train)
    sizes = groups.map(groups.value_counts()).to_numpy(dtype=float)
    weights = 1.0 / sizes
    weights /= weights.mean()
    return weights.astype(np.float32)


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}


def train_one(
    cfg: Config,
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    test_frames: dict[str, pd.DataFrame],
    seed: int,
    epochs: int,
    patience: int,
    output_dir: Path,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    seed_all(seed)
    pipeline = base.build_serializable_feature_pipeline()
    x_train = pipeline.fit_transform(train)
    x_validation = pipeline.transform(validation) if validation is not None else None
    x_tests = {name: pipeline.transform(frame) for name, frame in test_frames.items()}
    t_scaler = StandardScaler().fit(train[[TEMP]]) if cfg.family == "dnn" else None
    if t_scaler is not None:
        t_mean = torch.as_tensor(t_scaler.mean_, dtype=torch.float32, device=device).view(1, -1)
        t_scale = torch.as_tensor(t_scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    else:
        t_mean = t_scale = None
    ea_target, ea_mask, ea_details = empirical_ea_targets(train)
    weights = group_weights(train, cfg.group_balanced)
    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(train[TEMP].to_numpy(), dtype=torch.float32).view(-1, 1),
        torch.as_tensor(train[TARGET].to_numpy(), dtype=torch.float32).view(-1, 1),
        torch.as_tensor(weights, dtype=torch.float32).view(-1, 1),
        torch.as_tensor(ea_target, dtype=torch.float32).view(-1, 1),
        torch.as_tensor(ea_mask, dtype=torch.float32).view(-1, 1),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model: nn.Module = EnhancedPIML(x_train.shape[1], cfg) if cfg.family == "piml" else TunedDNN(x_train.shape[1], cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_epoch = -1
    best_mse = float("inf")
    stagnant = 0
    history = []
    total_epochs = fixed_epochs if fixed_epochs is not None else epochs

    def predict(x: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        model.eval()
        with torch.no_grad():
            features = torch.as_tensor(x, dtype=torch.float32, device=device)
            if cfg.family == "piml":
                temperature = torch.as_tensor(frame[TEMP].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
                output, ea, _ = model(features, temperature)
                return output.cpu().numpy().ravel(), ea.cpu().numpy().ravel()
            temperature_raw = torch.as_tensor(frame[TEMP].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
            temperature = (temperature_raw - t_mean) / t_scale
            return model(features, temperature).cpu().numpy().ravel(), None

    for epoch in range(total_epochs):
        model.train()
        losses = []
        for features, temperature, target, weight, target_ea, mask_ea in loader:
            features, temperature, target = features.to(device), temperature.to(device), target.to(device)
            weight, target_ea, mask_ea = weight.to(device), target_ea.to(device), mask_ea.to(device)
            optimizer.zero_grad()
            if cfg.family == "piml":
                pred, predicted_ea, _ = model(features, temperature)
            else:
                scaled = (temperature - t_mean) / t_scale
                pred, predicted_ea = model(features, scaled), None
            if cfg.loss == "huber":
                point_loss = nn.functional.smooth_l1_loss(pred, target, beta=0.2, reduction="none")
            else:
                point_loss = (pred - target) ** 2
            loss = (point_loss * weight).mean()
            if cfg.family == "piml" and cfg.ea_aux_weight > 0 and mask_ea.sum() > 0:
                auxiliary = (((predicted_ea - target_ea) ** 2) * weight * mask_ea).sum() / mask_ea.sum()
                loss = loss + cfg.ea_aux_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if validation is not None:
            val_pred, _ = predict(x_validation, validation)
            val_mse = float(mean_squared_error(validation[TARGET].to_numpy(), val_pred))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mse": val_mse})
            if val_mse < best_mse - 1e-8:
                best_mse, best_epoch, best_state, stagnant = val_mse, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stagnant += 1
            if stagnant >= patience:
                break
        else:
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_state, best_epoch = copy.deepcopy(model.state_dict()), total_epochs - 1
    predictions = {}
    test_metrics = {}
    for name, x_test in x_tests.items():
        pred, ea = predict(x_test, test_frames[name])
        predictions[name] = pred
        test_metrics[name] = {**score(test_frames[name][TARGET].to_numpy(), pred)}
        if ea is not None:
            test_metrics[name]["ea_median"] = float(np.median(ea))
            test_metrics[name]["ea_min"] = float(np.min(ea))
            test_metrics[name]["ea_max"] = float(np.max(ea))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pth"
    torch.save({"state_dict": best_state, "config": asdict(cfg), "input_dim": x_train.shape[1]}, checkpoint)
    joblib.dump(pipeline, output_dir / "preprocessor.joblib")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    result = {
        "config": cfg.name,
        "family": cfg.family,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "empirical_ea": ea_details,
        "test_metrics": test_metrics,
    }
    return result, predictions


def grouped_development_cv(
    dev: pd.DataFrame,
    configs: list[Config],
    output: Path,
    epochs: int,
    patience: int,
    device: torch.device,
) -> pd.DataFrame:
    groups = group_series(dev)
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=PARTITION_SEEDS["development_cv"])
    rows = []
    for cfg in configs:
        for fold, (train_idx, validation_idx) in enumerate(splitter.split(dev, groups=groups)):
            train = dev.iloc[train_idx].copy()
            validation = dev.iloc[validation_idx].copy()
            result, _ = train_one(
                cfg,
                train,
                validation,
                {"validation": validation},
                CONFIG_SEED,
                epochs,
                patience,
                output / "cv_models" / cfg.name / f"fold_{fold}",
                device,
            )
            metric = result["test_metrics"]["validation"]
            rows.append(
                {
                    "family": cfg.family,
                    "config": cfg.name,
                    "fold": fold,
                    "rmse": metric["rmse"],
                    "r2": metric["r2"],
                    "best_epoch": result["best_epoch"],
                    "epochs_ran": result["epochs_ran"],
                    "ea_aux_weight": cfg.ea_aux_weight,
                    "group_balanced": cfg.group_balanced,
                    "norm": cfg.norm,
                    "loss": cfg.loss,
                }
            )
            pd.DataFrame(rows).to_csv(output / "cv_trials.csv", index=False)
            print(f"[CV] {cfg.name} fold={fold} RMSE={metric['rmse']:.6f} R2={metric['r2']:.6f}", flush=True)
    frame = pd.DataFrame(rows)
    aggregation = (
        frame.groupby(["family", "config"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"), mean_r2=("r2", "mean"), median_best_epoch=("best_epoch", "median"))
        .sort_values(["family", "mean_rmse"])
    )
    aggregation.to_csv(output / "cv_config_summary.csv", index=False)
    return aggregation


def group_bootstrap_delta(test: pd.DataFrame, piml: np.ndarray, dnn: np.ndarray, seed: int = 42, n: int = 3000) -> dict:
    groups = group_series(test).to_numpy()
    unique = np.unique(groups)
    indices = {group: np.where(groups == group)[0] for group in unique}
    rng = np.random.default_rng(seed)
    delta = []
    y = test[TARGET].to_numpy()
    for _ in range(n):
        selected = rng.choice(unique, size=len(unique), replace=True)
        row_idx = np.concatenate([indices[group] for group in selected])
        p_rmse = np.sqrt(mean_squared_error(y[row_idx], piml[row_idx]))
        d_rmse = np.sqrt(mean_squared_error(y[row_idx], dnn[row_idx]))
        delta.append(d_rmse - p_rmse)
    values = np.array(delta)
    return {
        "delta_rmse_dnn_minus_piml": float(np.sqrt(mean_squared_error(y, dnn)) - np.sqrt(mean_squared_error(y, piml))),
        "bootstrap_ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "bootstrap_probability_piml_better": float(np.mean(values > 0)),
    }


def final_ensembles(
    dev: pd.DataFrame,
    confirmations: dict[str, pd.DataFrame],
    selected: dict[str, Config],
    selected_epochs: dict[str, int],
    output: Path,
    device: torch.device,
) -> dict:
    family_results = {}
    ensemble_predictions = {}
    for family, cfg in selected.items():
        seed_results = []
        prediction_banks = {name: [] for name in confirmations}
        for seed in FINAL_SEEDS:
            result, prediction = train_one(
                cfg,
                dev,
                None,
                confirmations,
                seed,
                selected_epochs[family],
                0,
                output / "final_models" / family / f"seed_{seed}",
                device,
                fixed_epochs=selected_epochs[family],
            )
            seed_results.append(result)
            for name in confirmations:
                prediction_banks[name].append(prediction[name])
            print(f"[FINAL] {family} seed={seed} trained epochs={selected_epochs[family]}", flush=True)
        metrics = {}
        for name, test in confirmations.items():
            pred = np.mean(prediction_banks[name], axis=0)
            ensemble_predictions[(family, name)] = pred
            metrics[name] = score(test[TARGET].to_numpy(), pred)
            predictions = test[["sample_id", TARGET, TEMP]].copy()
            predictions["prediction"] = pred
            predictions.to_csv(output / f"{family}_{name}_ensemble_predictions.csv", index=False)
        family_results[family] = {"config": asdict(cfg), "fixed_epochs": selected_epochs[family], "seeds": FINAL_SEEDS, "ensemble_metrics": metrics, "members": seed_results}
    pooled = pd.concat(confirmations.values(), ignore_index=True)
    piml_pooled = np.concatenate([ensemble_predictions[("piml", name)] for name in confirmations])
    dnn_pooled = np.concatenate([ensemble_predictions[("dnn", name)] for name in confirmations])
    comparison = {
        name: {
            "piml": family_results["piml"]["ensemble_metrics"][name],
            "dnn": family_results["dnn"]["ensemble_metrics"][name],
            "piml_beats_dnn_rmse": family_results["piml"]["ensemble_metrics"][name]["rmse"] < family_results["dnn"]["ensemble_metrics"][name]["rmse"],
            "piml_beats_dnn_r2": family_results["piml"]["ensemble_metrics"][name]["r2"] > family_results["dnn"]["ensemble_metrics"][name]["r2"],
        }
        for name in confirmations
    }
    comparison["pooled_confirmation"] = {
        "piml": score(pooled[TARGET].to_numpy(), piml_pooled),
        "dnn": score(pooled[TARGET].to_numpy(), dnn_pooled),
        **group_bootstrap_delta(pooled, piml_pooled, dnn_pooled),
    }
    return {"families": family_results, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=280)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--screen-limit-piml", type=int, default=0)
    parser.add_argument("--screen-limit-dnn", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = base.MaterialDataProcessor().load_and_preprocess_data_for_training_piml()
    parts = reserve_partitions(df)
    memberships = []
    for name, frame in parts.items():
        member = frame[["sample_id"]].copy()
        member["partition"] = name
        member["group_id"] = group_series(frame).values
        memberships.append(member)
    pd.concat(memberships, ignore_index=True).to_csv(output / "reserved_partition_membership.csv", index=False)
    configs = configurations()
    piml_configs = [cfg for cfg in configs if cfg.family == "piml"]
    dnn_configs = [cfg for cfg in configs if cfg.family == "dnn"]
    if args.screen_limit_piml:
        piml_configs = piml_configs[: args.screen_limit_piml]
    if args.screen_limit_dnn:
        dnn_configs = dnn_configs[: args.screen_limit_dnn]
    cv = grouped_development_cv(parts["development"], piml_configs + dnn_configs, output, args.epochs, args.patience, device)
    selected = {}
    selected_epochs = {}
    for family, config_list in [("piml", piml_configs), ("dnn", dnn_configs)]:
        best_name = cv[cv.family == family].sort_values("mean_rmse").iloc[0]["config"]
        selected[family] = next(cfg for cfg in config_list if cfg.name == best_name)
        median_epoch = int(cv[(cv.family == family) & (cv.config == best_name)]["median_best_epoch"].iloc[0]) + 1
        selected_epochs[family] = max(median_epoch, 30)
    confirmation = final_ensembles(
        parts["development"],
        {"confirmation_a": parts["confirmation_a"], "confirmation_b": parts["confirmation_b"]},
        selected,
        selected_epochs,
        output,
        device,
    )
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "design": {
            "selection": "3-fold grouped CV inside development partition only",
            "confirmation": "two reserved grouped partitions; fixed five-seed ensemble",
            "partition_seeds": PARTITION_SEEDS,
            "final_seeds": FINAL_SEEDS,
            "n_rows": {name: len(frame) for name, frame in parts.items()},
            "n_groups": {name: int(group_series(frame).nunique()) for name, frame in parts.items()},
        },
        "device": {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
        "selected": {family: {"config": asdict(cfg), "fixed_epochs": selected_epochs[family]} for family, cfg in selected.items()},
        "confirmation": confirmation,
        "guardrail": "The archived paper candidate checkpoint was not modified; inverse-design claims remain separate.",
    }
    (output / "campaign_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": summary["selected"], "comparison": confirmation["comparison"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
