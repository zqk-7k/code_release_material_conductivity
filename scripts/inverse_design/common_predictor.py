#!/usr/bin/env python3
from __future__ import annotations
import math, os, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

BASE = Path('/root/autodl-tmp/qkzhang/material-conductivity-reproduce')
ML = BASE / 'material-conductivity-data-analysis-ml'
sys.path.insert(0, str(ML / 'src' / 'zirconia'))
os.chdir(ML)

from config import path_config  # noqa
from etl.material_data_processor import MaterialDataProcessor  # noqa
from features.preprocessor import build_feature_pipeline  # noqa
from models.piml_net import PhysicsInformedNet  # noqa

DOPANTS_DB = {'Sc': 87.0, 'Yb': 98.5, 'Y': 101.9, 'Gd': 105.3, 'Sm': 107.9, 'Nd': 110.9, 'Ca': 112.0, 'Mg': 89.0}
VALENCE_DB = {k: 3.0 for k in DOPANTS_DB}
VALENCE_DB['Ca'] = 2.0
VALENCE_DB['Mg'] = 2.0
DOPANTS = list(DOPANTS_DB.keys())
PREDICTOR_VERSION = 'autonomous_common_predictor_v1'

@dataclass
class PredictionResult:
    dopant_1: str
    dopant_2: str
    f1_fraction: float
    f2_fraction: float
    f1_mol_percent: float
    f2_mol_percent: float
    total_dopant_fraction: float
    total_dopant_mol_percent: float
    sintering_temperature: float
    measurement_temperature: float
    predicted_log10_sigma: float
    predicted_sigma_s_cm: float
    constraint_passed: bool
    feature_range_warning: str
    model_checkpoint: str
    predictor_version: str
    dopant_average_radius_pm: float
    average_cation_radius_pm: float
    radius_mismatch_pm: float
    cation_anion_ratio: float
    average_dopant_valence: float
    plausible_region: str
    distance_to_training_scaled: float
    oxygen_vacancy_per_108_cations: int
    supercell_108_suggestion: str
    source: str = ''
    notes: str = ''

class CommonPredictor:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = str(checkpoint_path or path_config.BEST_PIML_MODEL_PATH)
        self.processor = MaterialDataProcessor()
        self.df = self.processor.load_and_preprocess_data_for_training_piml()
        self.pipeline = build_feature_pipeline()
        self.X = self.pipeline.fit_transform(self.df)
        self.template = self.df.iloc[0].copy()
        self.column_dtypes = self.df.dtypes
        self.model = PhysicsInformedNet(self.X.shape[1]).to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.eval()
        self.feature_ranges = self._feature_ranges()

    def _feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        out = {}
        for c in ['total_dopant_fraction', 'average_dopant_radius', 'average_dopant_valence', 'maximum_sintering_temperature']:
            s = pd.to_numeric(self.df[c], errors='coerce')
            out[c] = (float(s.min()), float(s.max()))
        return out

    @staticmethod
    def normalize_fraction(x: float) -> float:
        x = float(x)
        return x / 100.0 if x > 1.0 else x

    def physics_metrics(self, d1: str, d2: str, f1: float, f2: float):
        total = f1 + f2
        host = max(0.0, 1.0 - total)
        r1, r2 = DOPANTS_DB[d1], DOPANTS_DB[d2]
        v1, v2 = VALENCE_DB[d1], VALENCE_DB[d2]
        dop_avg = (r1*f1 + r2*f2) / total
        avg_val = (v1*f1 + v2*f2) / total
        avg_cat = host*84.0 + f1*r1 + f2*r2
        var = host*(84.0-avg_cat)**2 + f1*(r1-avg_cat)**2 + f2*(r2-avg_cat)**2
        mismatch = math.sqrt(max(0.0, var))
        ratio = avg_cat/138.0
        charge_deficit = f1*(4-v1) + f2*(4-v2)
        vac108 = int(round((charge_deficit/2.0) * 108))
        return dop_avg, avg_val, avg_cat, mismatch, ratio, vac108

    def distance_to_training(self, total, dop_avg, avg_val, temp):
        vals = {'total_dopant_fraction': total, 'average_dopant_radius': dop_avg, 'average_dopant_valence': avg_val, 'maximum_sintering_temperature': temp}
        dist = 0.0
        for k,v in vals.items():
            lo,hi = self.feature_ranges[k]
            span = max(hi-lo, 1e-9)
            if v < lo: dist += ((lo-v)/span)**2
            if v > hi: dist += ((v-hi)/span)**2
        return math.sqrt(dist)

    def plausible_region(self, avg_cat, mismatch):
        ratio = avg_cat/138.0
        if ratio >= 0.615 and mismatch < 6.5:
            return 'stable_like_ysz'
        if ratio >= 0.600 and mismatch < 9.0:
            return 'metastable_plausible'
        return 'higher_risk'

    def check(self, d1, d2, f1, f2, temp):
        warnings = []
        passed = True
        if d1 not in DOPANTS_DB or d2 not in DOPANTS_DB:
            passed = False; warnings.append('unknown dopant')
        if d1 == d2:
            passed = False; warnings.append('dopant_1 == dopant_2')
        total = f1 + f2
        if total < 0.08 or total > 0.20:
            passed = False; warnings.append(f'total dopant {total:.4f} outside 0.08-0.20')
        lo,hi = self.feature_ranges['maximum_sintering_temperature']
        if temp < lo or temp > hi:
            warnings.append(f'sintering temp {temp:.1f} outside training range {lo:.1f}-{hi:.1f}')
        return passed, '; '.join(warnings)

    def _make_df(self, candidates):
        rows = []
        for i,(d1,d2,f1,f2,temp) in enumerate(candidates):
            dop_avg, avg_val, *_ = self.physics_metrics(d1,d2,f1,f2)
            row = self.template.copy()
            row['sample_id'] = f'AUTO_{i}'
            row['material_source_and_purity'] = 'Autonomous inverse design'
            row['synthesis_method'] = 'Solid State Reaction'
            row['total_dopant_fraction'] = f1 + f2
            row['average_dopant_radius'] = dop_avg
            row['average_dopant_valence'] = avg_val
            row['number_of_dopants'] = 2
            row['maximum_sintering_temperature'] = temp
            row['primary_dopant_element'] = d1 if f1 >= f2 else d2
            rows.append(row)
        dfb = pd.DataFrame(rows)
        try: dfb = dfb.astype(self.column_dtypes)
        except Exception: pass
        return dfb

    def predict_batch(self, candidates, measurement_temperature=800, source='', notes='') -> List[PredictionResult]:
        candidates = [(d1,d2,self.normalize_fraction(f1),self.normalize_fraction(f2),float(temp)) for d1,d2,f1,f2,temp in candidates]
        work = candidates if len(candidates) > 1 else candidates + candidates
        dfb = self._make_df(work)
        Xv = self.pipeline.transform(dfb)
        Xt = torch.FloatTensor(Xv).to(self.device)
        T = torch.FloatTensor([[float(measurement_temperature)+273.15]] * len(work)).to(self.device)
        with torch.no_grad():
            pred, _, _ = self.model(Xt, T)
        preds = pred.detach().cpu().numpy().flatten().tolist()
        out = []
        for i,(d1,d2,f1,f2,temp) in enumerate(candidates):
            total = f1 + f2
            dop_avg, avg_val, avg_cat, mismatch, ratio, vac108 = self.physics_metrics(d1,d2,f1,f2)
            passed, warn = self.check(d1,d2,f1,f2,temp)
            n1, n2 = round(f1*108), round(f2*108)
            zr = 108 - n1 - n2
            suggestion = f'108 cation sites: Zr{zr} {d1}{n1} {d2}{n2}; actual {100*n1/108:.2f}/{100*n2/108:.2f} mol%; oxygen vacancies approx {vac108}'
            pv = float(preds[i])
            out.append(PredictionResult(d1,d2,f1,f2,f1*100,f2*100,total,total*100,temp,float(measurement_temperature),pv,10**pv,passed,warn,self.checkpoint_path,PREDICTOR_VERSION,dop_avg,avg_cat,mismatch,ratio,avg_val,self.plausible_region(avg_cat,mismatch),self.distance_to_training(total,dop_avg,avg_val,temp),vac108,suggestion,source,notes))
        return out

    def predict_candidate(self, dopant_1, dopant_2, f1, f2, sintering_temperature, measurement_temperature=800, checkpoint_path=None, return_features=True):
        return self.predict_batch([(dopant_1,dopant_2,f1,f2,sintering_temperature)], measurement_temperature)[0]

def result_to_dict(r: PredictionResult):
    return asdict(r)
