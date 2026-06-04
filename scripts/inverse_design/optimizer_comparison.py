#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

BASE = Path("/root/autodl-tmp/qkzhang/material-conductivity-reproduce")
ML = BASE / "material-conductivity-data-analysis-ml"
EXP = BASE / "optimizer_experiments"
RESULTS = EXP / "results"
LOG_DIR = BASE / "logs"
REPORT = EXP / "optimizer_comparison_report.md"
CN_REPORT = BASE / "优化器对比与候选筛选报告_中文.md"
RESULTS.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ML / "src" / "zirconia"))
os.chdir(ML)

from config import path_config  # noqa: E402
from etl.material_data_processor import MaterialDataProcessor  # noqa: E402
from features.preprocessor import build_feature_pipeline  # noqa: E402
from models.piml_net import PhysicsInformedNet  # noqa: E402
from algorithm.co_doping_ga import CoDopingGA  # noqa: E402

DOPANTS_DB = {
    "Sc": 87.0, "Yb": 98.5, "Y": 101.9, "Gd": 105.3,
    "Sm": 107.9, "Nd": 110.9, "Ca": 112.0, "Mg": 89.0,
}
VALENCE_DB = {k: 3.0 for k in DOPANTS_DB}
VALENCE_DB["Ca"] = 2.0
VALENCE_DB["Mg"] = 2.0
DOPANTS = list(DOPANTS_DB.keys())
LITERATURE_PRECEDENT = {"Sc-Mg", "Mg-Sc", "Sc-Y", "Y-Sc", "Y-Mg", "Mg-Y", "Sc-Yb", "Yb-Sc"}
RY_TO_EV = 13.605693122994

@dataclass
class CandidateResult:
    method: str
    seed: str
    candidate_name: str
    dopant_1: str
    dopant_2: str
    f1: float
    f2: float
    total_dopant: float
    sintering_temperature: float
    measurement_temperature: float
    predicted_log10_sigma: float
    predicted_sigma_s_cm: float
    dopant_average_radius_pm: float
    average_cation_radius_pm: float
    radius_mismatch_pm: float
    cation_anion_ratio: float
    average_dopant_valence: float
    constraints_passed: bool
    plausible_region: str
    distance_to_training_scaled: float
    literature_precedent: str
    warning: str
    source_file: str
    notes: str

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Predictor] device={self.device}")
        self.processor = MaterialDataProcessor()
        self.df = self.processor.load_and_preprocess_data_for_training_piml()
        self.pipeline = build_feature_pipeline()
        self.X = self.pipeline.fit_transform(self.df)
        self.model = PhysicsInformedNet(self.X.shape[1]).to(self.device)
        self.model.load_state_dict(torch.load(path_config.BEST_PIML_MODEL_PATH, map_location=self.device))
        self.model.eval()
        self.template = self.df.iloc[0].copy()
        self.column_dtypes = self.df.dtypes
        self.feature_ranges = self._feature_ranges()
        print(f"[Predictor] loaded {len(self.df)} rows, input_dim={self.X.shape[1]}")
        print(f"[Predictor] checkpoint={path_config.BEST_PIML_MODEL_PATH}")

    def _feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        cols = ["total_dopant_fraction", "average_dopant_radius", "average_dopant_valence", "maximum_sintering_temperature"]
        out = {}
        for c in cols:
            s = pd.to_numeric(self.df[c], errors="coerce")
            out[c] = (float(s.min()), float(s.max()))
        return out

    def _check_constraints(self, d1: str, d2: str, f1: float, f2: float, temp: float) -> Tuple[bool, str]:
        msgs = []
        if d1 not in DOPANTS_DB: msgs.append(f"unknown dopant_1={d1}")
        if d2 not in DOPANTS_DB: msgs.append(f"unknown dopant_2={d2}")
        if d1 == d2: msgs.append("dopants must be different")
        total = f1 + f2
        if total < 0.08 or total > 0.20: msgs.append(f"total dopant {total:.4f} outside 0.08-0.20")
        if f1 <= 0 or f2 <= 0: msgs.append("fractions must be positive")
        if temp < self.feature_ranges["maximum_sintering_temperature"][0] or temp > self.feature_ranges["maximum_sintering_temperature"][1]:
            msgs.append(f"sintering temperature {temp:.1f} outside training range {self.feature_ranges[maximum_sintering_temperature]}")
        return len(msgs) == 0, "; ".join(msgs)

    def _physics_metrics(self, d1: str, d2: str, f1: float, f2: float) -> Dict[str, float]:
        total = f1 + f2
        host_conc = max(0.0, 1.0 - total)
        r1, r2 = DOPANTS_DB[d1], DOPANTS_DB[d2]
        v1, v2 = VALENCE_DB[d1], VALENCE_DB[d2]
        dop_avg_r = (r1 * f1 + r2 * f2) / total
        avg_val = (v1 * f1 + v2 * f2) / total
        avg_cat_r = host_conc * 84.0 + f1 * r1 + f2 * r2
        variance = host_conc * (84.0 - avg_cat_r) ** 2 + f1 * (r1 - avg_cat_r) ** 2 + f2 * (r2 - avg_cat_r) ** 2
        mismatch = math.sqrt(max(0.0, variance))
        ratio = avg_cat_r / 138.0
        return {
            "dopant_average_radius_pm": dop_avg_r,
            "average_dopant_valence": avg_val,
            "average_cation_radius_pm": avg_cat_r,
            "radius_mismatch_pm": mismatch,
            "cation_anion_ratio": ratio,
        }

    def _plausible_region(self, avg_cat_r: float, mismatch: float) -> str:
        ratio = avg_cat_r / 138.0
        if ratio >= 0.615 and mismatch < 6.5:
            return "stable_like_ysz"
        if ratio >= 0.600 and mismatch < 9.0:
            return "metastable_plausible"
        return "higher_risk"

    def _distance_to_training(self, total: float, dop_avg_r: float, avg_val: float, temp: float) -> float:
        vals = {
            "total_dopant_fraction": total,
            "average_dopant_radius": dop_avg_r,
            "average_dopant_valence": avg_val,
            "maximum_sintering_temperature": temp,
        }
        dist = 0.0
        for k, v in vals.items():
            lo, hi = self.feature_ranges[k]
            span = max(hi - lo, 1e-9)
            if v < lo:
                dist += ((lo - v) / span) ** 2
            elif v > hi:
                dist += ((v - hi) / span) ** 2
        return math.sqrt(dist)

    def _make_df(self, candidates: List[Tuple[str, str, float, float, float]]) -> pd.DataFrame:
        rows = []
        for i, (d1, d2, f1, f2, temp) in enumerate(candidates):
            phys = self._physics_metrics(d1, d2, f1, f2)
            row = self.template.copy()
            row["sample_id"] = f"Opt_{i}"
            row["material_source_and_purity"] = "Optimizer Experiment Co-Doping"
            row["synthesis_method"] = "Solid State Reaction"
            row["total_dopant_fraction"] = f1 + f2
            row["average_dopant_radius"] = phys["dopant_average_radius_pm"]
            row["average_dopant_valence"] = phys["average_dopant_valence"]
            row["number_of_dopants"] = 2
            row["maximum_sintering_temperature"] = temp
            row["primary_dopant_element"] = d1 if f1 >= f2 else d2
            rows.append(row)
        dfb = pd.DataFrame(rows)
        try:
            dfb = dfb.astype(self.column_dtypes)
        except Exception:
            pass
        return dfb

    def predict_batch(self, candidates: List[Tuple[str, str, float, float, float]], measurement_temperature: float = 800.0,
                      method: str = "", seed: str = "", names: Optional[List[str]] = None,
                      source_file: str = "", notes: str = "") -> List[CandidateResult]:
        if not candidates:
            return []
        orig_n = len(candidates)
        work = candidates if len(candidates) > 1 else candidates + [candidates[0]]
        dfb = self._make_df(work)
        try:
            X_vec = self.pipeline.transform(dfb)
            X_tensor = torch.FloatTensor(X_vec).to(self.device)
            T = torch.FloatTensor([[measurement_temperature + 273.15]] * len(work)).to(self.device)
            with torch.no_grad():
                pred, _, _ = self.model(X_tensor, T)
            preds = pred.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"[WARN] prediction failed for batch size {len(work)}: {e}")
            preds = [-999.0] * len(work)
        out = []
        for i, (d1, d2, f1, f2, temp) in enumerate(candidates):
            predv = float(preds[i])
            phys = self._physics_metrics(d1, d2, f1, f2)
            passed, warn = self._check_constraints(d1, d2, f1, f2, temp)
            total = f1 + f2
            pair = f"{d1}-{d2}"
            out.append(CandidateResult(
                method=method,
                seed=str(seed),
                candidate_name=names[i] if names else f"{method}_{i}",
                dopant_1=d1,
                dopant_2=d2,
                f1=f1,
                f2=f2,
                total_dopant=total,
                sintering_temperature=temp,
                measurement_temperature=measurement_temperature,
                predicted_log10_sigma=predv,
                predicted_sigma_s_cm=10 ** predv if predv > -100 else 0.0,
                dopant_average_radius_pm=phys["dopant_average_radius_pm"],
                average_cation_radius_pm=phys["average_cation_radius_pm"],
                radius_mismatch_pm=phys["radius_mismatch_pm"],
                cation_anion_ratio=phys["cation_anion_ratio"],
                average_dopant_valence=phys["average_dopant_valence"],
                constraints_passed=passed,
                plausible_region=self._plausible_region(phys["average_cation_radius_pm"], phys["radius_mismatch_pm"]),
                distance_to_training_scaled=self._distance_to_training(total, phys["dopant_average_radius_pm"], phys["average_dopant_valence"], temp),
                literature_precedent="yes" if pair in LITERATURE_PRECEDENT else "unknown",
                warning=warn,
                source_file=source_file,
                notes=notes,
            ))
        return out

    def predict_candidate(self, dopant_1, dopant_2, f1, f2, sintering_temperature, measurement_temperature=800.0, **kwargs):
        f1 = normalize_fraction(f1)
        f2 = normalize_fraction(f2)
        candidate_name = kwargs.pop("candidate_name", None)
        names = [candidate_name] if candidate_name else None
        return self.predict_batch([(dopant_1, dopant_2, f1, f2, sintering_temperature)], measurement_temperature, names=names, **kwargs)[0]

def normalize_fraction(x: float) -> float:
    x = float(x)
    return x / 100.0 if x > 1 else x

def write_csv(path: Path, rows: List[CandidateResult], extra_fields: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(CandidateResult.__annotations__.keys())
    if extra_fields:
        fields += extra_fields
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

def topn(rows: List[CandidateResult], n=100) -> List[CandidateResult]:
    return sorted(rows, key=lambda r: r.predicted_log10_sigma, reverse=True)[:n]

def load_historical_candidates() -> List[Tuple[str, str, str, float, float, float, str, str]]:
    out = []
    files = [ML / "results/ai_discovery_best_recipe.csv"]
    files += sorted(BASE.glob("backup_before_full_reproduce*/results/ai_discovery_best_recipe.csv"))
    seen = set()
    for p in files:
        if not p.exists():
            continue
        with p.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    d1, d2 = row["dopant_1_element"], row["dopant_2_element"]
                    f1, f2 = float(row["dopant_1_fraction"]), float(row["dopant_2_fraction"])
                    t = float(row.get("sintering_temperature_c") or 1505)
                    key = (d1, d2, round(f1, 6), round(f2, 6), round(t, 2), str(p))
                    if key not in seen:
                        seen.add(key)
                        out.append((f"historical_{len(out)+1}", d1, d2, f1, f2, t, str(p), "historical/current ai_discovery_best_recipe.csv"))
                except Exception:
                    pass
    return out

def direct_evaluation(pred: Predictor) -> List[CandidateResult]:
    current_t = 1505.0
    current_file = ML / "results/ai_discovery_best_recipe.csv"
    if current_file.exists():
        with current_file.open(newline="") as f:
            row = next(csv.DictReader(f), None)
            if row:
                current_t = float(row.get("sintering_temperature_c") or current_t)
    items = [
        ("paper_candidate_Sc7p50_Mg3p19", "Sc", "Mg", 0.0750, 0.0319, 1505.0, "paper target in user prompt", "paper candidate direct check"),
        ("current_reproduced_GA", "Sc", "Mg", 0.0627, 0.0497, current_t, str(current_file), "rounded current reproduced GA candidate"),
    ]
    for name, d1, d2, f1, f2, t, src, note in load_historical_candidates():
        items.append((name, d1, d2, f1, f2, t, src, note))
    candidates = [(d1, d2, f1, f2, t) for _, d1, d2, f1, f2, t, _, _ in items]
    names = [x[0] for x in items]
    rows = pred.predict_batch(candidates, 800, method="direct", seed="NA", names=names, source_file=";".join(sorted(set(x[6] for x in items))), notes="direct evaluation")
    # adjust source/notes per row
    for r, item in zip(rows, items):
        r.source_file = item[6]
        r.notes = item[7]
    write_csv(RESULTS / "direct_candidate_evaluation.csv", rows)
    return rows

def original_ga_multi_seed(pred: Predictor) -> Tuple[List[CandidateResult], List[dict]]:
    seeds = [0, 1, 2, 3, 4, 42, 2026]
    best_rows = []
    history_rows = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        ga = CoDopingGA(pred.model, pred.pipeline, pred.df, device=pred.device)
        start = time.time()
        best_ind, best_score, history = ga.run(generations=25, population_size=60)
        dur = time.time() - start
        d1, f1, d2, f2, t = best_ind
        row = pred.predict_candidate(d1, d2, f1, f2, t, 800, method="original_ga", seed=str(seed), candidate_name=f"original_ga_seed_{seed}", source_file="co_doping_ga.py", notes=f"ga_reported_score={best_score:.6f}; duration_sec={dur:.2f}")
        best_rows.append(row)
        for gen, score in enumerate(history):
            history_rows.append({"seed": seed, "generation": gen, "best_score": score})
    write_csv(RESULTS / "original_ga_multi_seed.csv", best_rows)
    with (RESULTS / "original_ga_convergence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "generation", "best_score"])
        w.writeheader(); w.writerows(history_rows)
    return best_rows, history_rows

def random_search(pred: Predictor, n=20000, seed=2026) -> List[CandidateResult]:
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n):
        d1, d2 = rng.choice(DOPANTS, size=2, replace=False).tolist()
        # rejection for total constraint
        for _attempt in range(50):
            f1 = float(rng.uniform(0.04, 0.10))
            f2 = float(rng.uniform(0.01, 0.08))
            if 0.08 <= f1 + f2 <= 0.20:
                break
        t = float(rng.uniform(1300, 1600))
        candidates.append((d1, d2, f1, f2, t))
    rows = []
    batch = 2048
    for i in range(0, len(candidates), batch):
        rows.extend(pred.predict_batch(candidates[i:i+batch], 800, method="random_search", seed=str(seed), source_file="optimizer_comparison.py", notes=f"n={n}"))
    tops = topn(rows, 100)
    write_csv(RESULTS / "random_search_top100.csv", tops)
    return rows

def grid_search(pred: Predictor) -> Tuple[List[CandidateResult], List[CandidateResult]]:
    all_rows = []
    pair_best = []
    coarse_candidates = []
    pair_for_candidate = []
    f1_vals = np.arange(0.04, 0.1001, 0.005)
    f2_vals = np.arange(0.01, 0.0801, 0.005)
    t_vals = np.arange(1300, 1600.1, 50)
    pairs = [(a, b) for a in DOPANTS for b in DOPANTS if a != b]
    for d1, d2 in pairs:
        for f1 in f1_vals:
            for f2 in f2_vals:
                if not (0.08 <= f1 + f2 <= 0.20):
                    continue
                for t in t_vals:
                    coarse_candidates.append((d1, d2, float(f1), float(f2), float(t)))
                    pair_for_candidate.append((d1, d2))
    coarse_rows = []
    for i in range(0, len(coarse_candidates), 4096):
        coarse_rows.extend(pred.predict_batch(coarse_candidates[i:i+4096], 800, method="coarse_grid", seed="NA", source_file="optimizer_comparison.py", notes="f step 0.5 mol%, temp step 50 C"))
    all_rows.extend(coarse_rows)
    # top pairs by best coarse score
    best_by_pair: Dict[Tuple[str, str], CandidateResult] = {}
    for r in coarse_rows:
        k = (r.dopant_1, r.dopant_2)
        if k not in best_by_pair or r.predicted_log10_sigma > best_by_pair[k].predicted_log10_sigma:
            best_by_pair[k] = r
    top_pair_rows = topn(list(best_by_pair.values()), 10)
    fine_candidates = []
    for br in top_pair_rows:
        for f1 in np.arange(max(0.001, br.f1 - 0.01), br.f1 + 0.0101, 0.001):
            for f2 in np.arange(max(0.001, br.f2 - 0.01), br.f2 + 0.0101, 0.001):
                if not (0.08 <= f1 + f2 <= 0.20):
                    continue
                for t in np.arange(max(1000, br.sintering_temperature - 50), br.sintering_temperature + 50.1, 10):
                    fine_candidates.append((br.dopant_1, br.dopant_2, float(f1), float(f2), float(t)))
    fine_rows = []
    for i in range(0, len(fine_candidates), 4096):
        fine_rows.extend(pred.predict_batch(fine_candidates[i:i+4096], 800, method="fine_grid", seed="NA", source_file="optimizer_comparison.py", notes="top 10 pair local fine grid f step 0.1 mol%, temp step 10 C"))
    all_rows.extend(fine_rows)
    # final pair best
    best_by_pair = {}
    for r in all_rows:
        k = tuple(sorted([r.dopant_1, r.dopant_2]))
        if k not in best_by_pair or r.predicted_log10_sigma > best_by_pair[k].predicted_log10_sigma:
            best_by_pair[k] = r
    pair_best = topn(list(best_by_pair.values()), 1000)
    write_csv(RESULTS / "grid_search_top100.csv", topn(all_rows, 100))
    write_csv(RESULTS / "top_by_dopant_pair.csv", pair_best)
    return all_rows, pair_best

def sc_mg_local_search(pred: Predictor) -> List[CandidateResult]:
    rows = []
    candidates = []
    for f1 in np.arange(0.05, 0.10001, 0.0025):
        for f2 in np.arange(0.01, 0.06001, 0.0025):
            if 0.08 <= f1 + f2 <= 0.20:
                for t in np.arange(1400, 1600.1, 25):
                    candidates.append(("Sc", "Mg", float(f1), float(f2), float(t)))
    coarse = []
    for i in range(0, len(candidates), 4096):
        coarse.extend(pred.predict_batch(candidates[i:i+4096], 800, method="sc_mg_local_coarse", seed="NA", source_file="optimizer_comparison.py", notes="Sc-Mg local coarse grid"))
    best = topn(coarse, 1)[0]
    fine_candidates = []
    for f1 in np.arange(max(0.001, best.f1 - 0.01), best.f1 + 0.0101, 0.001):
        for f2 in np.arange(max(0.001, best.f2 - 0.01), best.f2 + 0.0101, 0.001):
            if 0.08 <= f1 + f2 <= 0.20:
                for t in np.arange(max(1200, best.sintering_temperature - 50), best.sintering_temperature + 50.1, 5):
                    fine_candidates.append(("Sc", "Mg", float(f1), float(f2), float(t)))
    fine = []
    for i in range(0, len(fine_candidates), 4096):
        fine.extend(pred.predict_batch(fine_candidates[i:i+4096], 800, method="sc_mg_local_fine", seed="NA", source_file="optimizer_comparison.py", notes="Sc-Mg local fine grid"))
    rows = coarse + fine
    write_csv(RESULTS / "sc_mg_local_surface.csv", rows)
    return rows

def ensure_optuna():
    try:
        import optuna  # noqa
        return True, "already installed"
    except Exception:
        print("[Optuna] not installed; trying pip install optuna")
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "optuna"], text=True).returncode
        if rc != 0:
            return False, f"pip install failed rc={rc}"
        try:
            import optuna  # noqa
            return True, "installed"
        except Exception as e:
            return False, repr(e)

def optuna_tpe(pred: Predictor, trials=3000, seeds=(0,1,2,3,4)) -> Tuple[List[CandidateResult], str]:
    ok, msg = ensure_optuna()
    if not ok:
        print(f"[Optuna] skipped: {msg}")
        (RESULTS / "optuna_tpe_results.csv").write_text(f"skipped_reason\n{msg}\n")
        return [], msg
    import optuna
    all_rows = []
    for seed in seeds:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        def objective(trial):
            d1 = trial.suggest_categorical("dopant_1", DOPANTS)
            d2 = trial.suggest_categorical("dopant_2", DOPANTS)
            if d1 == d2:
                return -999.0
            f1 = trial.suggest_float("f1", 0.04, 0.10)
            f2 = trial.suggest_float("f2", 0.01, 0.08)
            if not (0.08 <= f1 + f2 <= 0.20):
                return -999.0
            t = trial.suggest_float("sintering_temperature", 1300, 1600)
            res = pred.predict_candidate(d1, d2, f1, f2, t, 800, method="optuna_tpe", seed=str(seed), candidate_name=f"optuna_seed_{seed}_trial_{trial.number}", source_file="optimizer_comparison.py", notes=f"trials={trials}")
            trial.set_user_attr("candidate", asdict(res))
            return res.predicted_log10_sigma
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        trials_sorted = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)[:100]
        for t in trials_sorted:
            cand = t.user_attrs.get("candidate")
            if cand:
                all_rows.append(CandidateResult(**cand))
    write_csv(RESULTS / "optuna_tpe_results.csv", topn(all_rows, 500) if all_rows else [])
    return all_rows, msg

def differential_evolution_search(pred: Predictor, pair_rows: List[CandidateResult]) -> List[CandidateResult]:
    try:
        from scipy.optimize import differential_evolution
    except Exception as e:
        print(f"[DE] skipped: {e}")
        return []
    pairs = []
    for r in pair_rows:
        pair = (r.dopant_1, r.dopant_2)
        if pair not in pairs:
            pairs.append(pair)
        if len(pairs) >= 5:
            break
    out = []
    for d1, d2 in pairs:
        def obj(x):
            f1, f2, t = x
            if not (0.08 <= f1 + f2 <= 0.20):
                return 999
            return -pred.predict_candidate(d1, d2, f1, f2, t, 800, method="differential_evolution", seed="2026", candidate_name=f"de_{d1}_{d2}").predicted_log10_sigma
        res = differential_evolution(obj, bounds=[(0.04,0.10),(0.01,0.08),(1300,1600)], seed=2026, maxiter=40, popsize=10, polish=True, workers=1)
        f1, f2, t = res.x
        out.append(pred.predict_candidate(d1, d2, float(f1), float(f2), float(t), 800, method="differential_evolution", seed="2026", candidate_name=f"de_{d1}_{d2}", source_file="optimizer_comparison.py", notes=f"fun={res.fun}"))
    write_csv(RESULTS / "differential_evolution_results.csv", out)
    return out

def supercell_suggestion(r: CandidateResult, cation_sites=32) -> str:
    n1 = round(r.f1 * cation_sites)
    n2 = round(r.f2 * cation_sites)
    total = n1 + n2
    return f"For {cation_sites} cation sites: {r.dopant_1}{n1}, {r.dopant_2}{n2}, Zr{cation_sites-total}; fractions approx {100*n1/cation_sites:.2f}%/{100*n2/cation_sites:.2f}%"

def build_reports(all_rows: List[CandidateResult], direct_rows: List[CandidateResult], ga_rows: List[CandidateResult], pair_best: List[CandidateResult], optuna_msg: str):
    top_all = topn(all_rows, 100)
    write_csv(RESULTS / "top100_all_methods.csv", top_all)
    # comparison by method best
    method_best = {}
    for r in all_rows:
        if r.method not in method_best or r.predicted_log10_sigma > method_best[r.method].predicted_log10_sigma:
            method_best[r.method] = r
    comparison = topn(list(method_best.values()), 100)
    write_csv(RESULTS / "optimizer_comparison.csv", comparison)

    best = top_all[0]
    scmg = [r for r in all_rows if {r.dopant_1, r.dopant_2} == {"Sc", "Mg"}]
    best_scmg = topn(scmg, 1)[0] if scmg else None
    plausible = [r for r in all_rows if r.plausible_region in {"stable_like_ysz", "metastable_plausible"}]
    best_stable = topn(plausible, 1)[0] if plausible else best
    paper = next((r for r in direct_rows if r.candidate_name.startswith("paper_candidate")), None)
    current = next((r for r in direct_rows if r.candidate_name == "current_reproduced_GA"), None)
    closest = min(all_rows, key=lambda r: (r.dopant_1 != "Sc" or r.dopant_2 != "Mg", abs(r.f1-0.075)+abs(r.f2-0.0319)+abs(r.sintering_temperature-1505)/1000))
    recommended = []
    for r in [best, best_scmg, best_stable, closest, paper, current]:
        if r and all((r.candidate_name != x.candidate_name or r.method != x.method) for x in recommended):
            recommended.append(r)
    for r in recommended:
        r.notes = (r.notes + "; " if r.notes else "") + supercell_suggestion(r)
    write_csv(RESULTS / "recommended_candidates_for_md_dft.csv", recommended)

    summary = {
        "completed": True,
        "best_by_predicted_log10_sigma": asdict(best),
        "best_sc_mg_candidate": asdict(best_scmg) if best_scmg else None,
        "best_stability_filtered_candidate": asdict(best_stable),
        "closest_to_paper_candidate": asdict(closest),
        "paper_candidate_direct": asdict(paper) if paper else None,
        "current_ga_direct": asdict(current) if current else None,
        "new_best_improvement_vs_current_ga_log10": best.predicted_log10_sigma - current.predicted_log10_sigma if current else None,
        "new_best_difference_vs_paper_target_minus_1p036": best.predicted_log10_sigma - (-1.036),
        "optuna_status": optuna_msg,
        "result_files": [str(p) for p in sorted(RESULTS.glob("*.csv"))],
    }
    (RESULTS / "optimizer_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def md_row(r: CandidateResult) -> str:
        return f"{r.method} | {r.dopant_1}-{r.dopant_2} | {r.f1*100:.2f}% | {r.f2*100:.2f}% | {r.total_dopant*100:.2f}% | {r.sintering_temperature:.1f} | {r.predicted_log10_sigma:.4f} | {r.plausible_region}"
    lines = []
    lines.append("# Optimizer Comparison Report")
    lines.append("")
    lines.append("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("## Current GA Analysis")
    lines.append("")
    lines.append("- Variables: dopant_1, dopant_2, dopant_1_fraction, dopant_2_fraction, sintering_temperature.")
    lines.append("- Bounds: dopant elements from Sc/Yb/Y/Gd/Sm/Nd/Ca/Mg; f1 4-10 mol%, f2 1-8 mol%, sintering 1300-1600 C.")
    lines.append("- Constraints: total dopant fraction 8-20 mol%, dopant_1 != dopant_2.")
    lines.append("- Random seed: discovery script sets Python random, NumPy, and torch seed to 42; multi-seed experiment tests 0,1,2,3,4,42,2026.")
    lines.append(f"- Checkpoint: `{path_config.BEST_PIML_MODEL_PATH}`.")
    lines.append("- Prediction function: `CoDopingGA.calculate_population_fitness`, using project pipeline.transform and PhysicsInformedNet forward pass.")
    lines.append("- 800 C encoding: measurement temperature is target_temp_c + 273.15 = 1073.15 K tensor fed to the model.")
    lines.append("- Objective: single-objective maximize predicted log10 conductivity.")
    lines.append("- No stability/radius-mismatch/uncertainty penalty is included in the GA objective; those are added here only as post-screening metrics.")
    lines.append("- Why it may miss the paper candidate: small stochastic search budget, mixed categorical/continuous variables, retrained checkpoint prediction surface, no local refinement, no constraint-aware Bayesian sampler, and possible dependence on fitted feature pipeline/template row.")
    lines.append("")
    lines.append("## Direct Candidate Evaluation")
    lines.append("")
    lines.append("| Candidate | Pair | f1 | f2 | Temp C | predicted log10 sigma | Notes |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in direct_rows:
        lines.append(f"| {r.candidate_name} | {r.dopant_1}-{r.dopant_2} | {r.f1*100:.2f}% | {r.f2*100:.2f}% | {r.sintering_temperature:.1f} | {r.predicted_log10_sigma:.4f} | {r.notes} |")
    lines.append("")
    lines.append("## Best By Method")
    lines.append("")
    lines.append("| Method | Pair | f1 | f2 | Total | Sinter C | log10 sigma | Region |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in comparison:
        lines.append("| " + md_row(r) + " |")
    lines.append("")
    lines.append("## Recommended Candidates")
    lines.append("")
    lines.append("| Role | Method | Pair | f1 | f2 | Total | Sinter C | log10 sigma | Supercell suggestion |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    roles = ["best_by_predicted_log10_sigma", "best_sc_mg_candidate", "best_stability_filtered_candidate", "closest_to_paper_candidate", "paper_candidate_direct", "current_ga_direct"]
    for role, r in zip(roles, recommended):
        lines.append(f"| {role} | {r.method} | {r.dopant_1}-{r.dopant_2} | {r.f1*100:.2f}% | {r.f2*100:.2f}% | {r.total_dopant*100:.2f}% | {r.sintering_temperature:.1f} | {r.predicted_log10_sigma:.4f} | {supercell_suggestion(r)} |")
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")

    cn = []
    cn.append("# 优化器对比与候选筛选报告")
    cn.append("")
    cn.append("生成时间：" + time.strftime("%Y-%m-%d %H:%M:%S"))
    cn.append("")
    cn.append("## 1. 结论摘要")
    cn.append("")
    cn.append(f"当前 checkpoint 对论文候选 Sc=7.50 mol%、Mg=3.19 mol%、1505 C 的预测 log10σ 为 {paper.predicted_log10_sigma:.4f}。")
    cn.append(f"当前复现 GA 候选的直接预测 log10σ 为 {current.predicted_log10_sigma:.4f}。")
    cn.append(f"本轮优化器找到的最高 predicted_log10_sigma 为 {best.predicted_log10_sigma:.4f}，候选为 {best.dopant_1}-{best.dopant_2}: {best.f1*100:.2f}/{best.f2*100:.2f} mol%，烧结温度 {best.sintering_temperature:.1f} C。")
    cn.append(f"相对当前 GA 候选提升 {summary['new_best_improvement_vs_current_ga_log10']:+.4f} log10 单位；相对论文目标 -1.036 的差值为 {summary['new_best_difference_vs_paper_target_minus_1p036']:+.4f}。")
    cn.append("")
    cn.append("## 2. 必答问题")
    cn.append("")
    paper_judgement = "能/接近" if abs(paper.predicted_log10_sigma + 1.036) <= 0.05 else f"不能，当前直接预测为 {paper.predicted_log10_sigma:.4f}"
    cn.append(f"1. 当前 PIML checkpoint 是否能在论文候选附近预测出 -1.036：{paper_judgement}。")
    ga_judgement = "不是" if best.predicted_log10_sigma > current.predicted_log10_sigma + 1e-6 else "在本轮搜索中未发现更优"
    cn.append(f"2. 当前 GA 输出是否真是当前搜索空间最优：{ga_judgement}；本轮最优为 {best.method}。")
    cn.append(f"3. 原 GA、Random Search、Grid Search、Optuna TPE 中最高者：{best.method}，log10σ={best.predicted_log10_sigma:.4f}。")
    best_is_scmg = ({best.dopant_1, best.dopant_2} == {"Sc", "Mg"})
    cn.append(f"4. 最优结果是否仍是 Sc-Mg：{'是' if best_is_scmg else '不是，是 '+best.dopant_1+'-'+best.dopant_2}。")
    cn.append("5. 如果不是 Sc-Mg，论文一致性上仍建议保留 Sc-Mg 作为重点候选，因为 Sc-Mg 有实验/论文目标先例，但应将新最优作为额外候选。")
    cn.append(f"6. 新优化器最佳 predicted_log10_sigma：{best.predicted_log10_sigma:.4f}。")
    cn.append(f"7. 新候选比当前 GA 候选提高：{summary['new_best_improvement_vs_current_ga_log10']:+.4f} log10 单位。")
    cn.append(f"8. 新候选比论文候选 -1.036：{summary['new_best_difference_vs_paper_target_minus_1p036']:+.4f}，{'更高' if summary['new_best_difference_vs_paper_target_minus_1p036']>0 else '更低'}。")
    cn.append("9. 若新候选偏离论文候选，可能原因包括 checkpoint 变化、预测面变化、旧 GA 随机性、搜索边界差异、目标函数未包含稳定性/不确定性、以及特征构造依赖模板样本。")
    cn.append("10. 不建议直接把论文中的 GA 结果替换为新候选；建议作为当前 checkpoint 下的新逆向设计结果，并保留论文 Sc-Mg 候选用于对照。")
    cn.append("11. DFT/CHGNet 是否改用新配比：建议同时保留论文 Sc-Mg 和当前最优候选进入后续计算筛选，而不是只替换。")
    cn.append(f"12. integer supercell 建议：最佳候选：{supercell_suggestion(best)}；Sc-Mg 最佳：{supercell_suggestion(best_scmg) if best_scmg else '无'}。")
    cn.append("## 3. 方法最佳结果")
    cn.append("")
    cn.append("| 方法 | 元素 | f1 | f2 | total | sinter C | log10σ | 区域 |")
    cn.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in comparison:
        cn.append(f"| {r.method} | {r.dopant_1}-{r.dopant_2} | {r.f1*100:.2f}% | {r.f2*100:.2f}% | {r.total_dopant*100:.2f}% | {r.sintering_temperature:.1f} | {r.predicted_log10_sigma:.4f} | {r.plausible_region} |")
    cn.append("")
    cn.append("## 4. 结果文件")
    cn.append("")
    for p in sorted(RESULTS.glob("*.csv")):
        cn.append(f"- `{p}`")
    cn.append(f"- `{RESULTS / 'optimizer_comparison_summary.json'}`")
    cn.append(f"- `{REPORT}`")
    cn.append("")
    cn.append("本阶段没有重新训练 PIML/DNN/RF/XGBoost，没有运行 CHGNet MD，没有运行 DFT/QE，没有覆盖原始 `results/ai_discovery_best_recipe.csv`。")
    CN_REPORT.write_text("\n".join(cn) + "\n")

    return summary

def main():
    print("========== Optimizer comparison started ==========")
    print(f"BASE={BASE}")
    print(f"ML={ML}")
    pred = Predictor()
    direct_rows = direct_evaluation(pred)
    all_rows = list(direct_rows)
    ga_rows, _ = original_ga_multi_seed(pred)
    all_rows.extend(ga_rows)
    random_rows = random_search(pred, n=20000, seed=2026)
    all_rows.extend(random_rows)
    grid_rows, pair_best = grid_search(pred)
    all_rows.extend(grid_rows)
    scmg_rows = sc_mg_local_search(pred)
    all_rows.extend(scmg_rows)
    optuna_rows, optuna_msg = optuna_tpe(pred, trials=3000, seeds=(0,1,2,3,4))
    all_rows.extend(optuna_rows)
    de_rows = differential_evolution_search(pred, pair_best)
    all_rows.extend(de_rows)
    summary = build_reports(all_rows, direct_rows, ga_rows, pair_best, optuna_msg)
    print("========== Optimizer comparison completed ==========")
    print(json.dumps({
        "best_method": summary["best_by_predicted_log10_sigma"]["method"],
        "best_pair": summary["best_by_predicted_log10_sigma"]["dopant_1"] + "-" + summary["best_by_predicted_log10_sigma"]["dopant_2"],
        "best_log10_sigma": summary["best_by_predicted_log10_sigma"]["predicted_log10_sigma"],
        "summary_json": str(RESULTS / "optimizer_comparison_summary.json"),
        "cn_report": str(CN_REPORT),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
