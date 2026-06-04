#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os, random, shutil, subprocess, sys, time, traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

BASE = Path('/root/autodl-tmp/qkzhang/material-conductivity-reproduce')
ML = BASE / 'material-conductivity-data-analysis-ml'
AID = BASE / 'autonomous_inverse_design'
RESULTS = AID / 'results'
SCRIPTS = AID / 'scripts'
LOGDIR = BASE / 'logs' / 'autonomous_inverse_design'
MASTER_LOG = LOGDIR / 'autonomous_master.log'
CN_REPORT = BASE / '自主逆向设计优化报告_中文.md'
TECH_REPORT = AID / 'autonomous_inverse_design_report.md'
STATUS_CSV = RESULTS / 'stage_status.csv'
STATUS_JSON = RESULTS / 'stage_status.json'
for p in [RESULTS,SCRIPTS,LOGDIR]: p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPTS))
from common_predictor import CommonPredictor, DOPANTS, DOPANTS_DB, result_to_dict

PAPER_TARGET = -1.036
STOP_TARGET = -1.036
FINE_GRID_BASELINE = -1.3399
CURRENT_GA_BASELINE = -1.4655
ALL_CANDIDATES: List[dict] = []
STAGES: List[dict] = []
BEST = None

class Tee:
    def __init__(self, *paths):
        self.files = [open(p, 'a', encoding='utf-8') for p in paths]
    def write(self, text):
        sys.__stdout__.write(text); sys.__stdout__.flush()
        for f in self.files: f.write(text); f.flush()
    def flush(self):
        sys.__stdout__.flush()
        for f in self.files: f.flush()

def now(): return time.strftime('%Y-%m-%d %H:%M:%S')
def append(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f: f.write(text)
def write_json(path: Path, obj): path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def best_from_rows(rows):
    valid = [r for r in rows if r.get('constraint_passed')]
    return max(valid or rows, key=lambda r: r.get('predicted_log10_sigma', -999)) if rows else None

def update_best(rows):
    global BEST
    b = best_from_rows(rows)
    if b and (BEST is None or b['predicted_log10_sigma'] > BEST['predicted_log10_sigma']):
        BEST = b
    return BEST

def save_rows(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def record_stage(name, start, end, success, command, log_path, result_files, error='', next_decision=''):
    best = BEST or {}
    rec = {
        'stage': name, 'start': start, 'end': end, 'success': bool(success), 'command': command,
        'log_path': str(log_path), 'result_files': ';'.join(map(str,result_files)), 'error': error,
        'best_candidate': f"{best.get('dopant_1','')}-{best.get('dopant_2','')} {best.get('f1_mol_percent','')} {best.get('f2_mol_percent','')}",
        'best_predicted_log10_sigma': best.get('predicted_log10_sigma'),
        'gap_to_paper_minus_1p036': None if not best else best.get('predicted_log10_sigma') - PAPER_TARGET,
        'next_decision': next_decision,
    }
    STAGES.append(rec)
    save_rows(STATUS_CSV, STAGES); write_json(STATUS_JSON, STAGES)
    block = f"\n## 阶段：{name}\n\n- 开始：{start}\n- 结束：{end}\n- 成功：{success}\n- 命令：`{command}`\n- 日志：`{log_path}`\n- 结果文件：{', '.join(map(str,result_files))}\n- 当前最佳候选：{rec['best_candidate']}\n- 当前 best predicted_log10_sigma：{rec['best_predicted_log10_sigma']}\n- 与论文 -1.036 差距：{rec['gap_to_paper_minus_1p036']}\n- 下一步自动决策：{next_decision}\n"
    if error: block += f"- 错误：`{error}`\n"
    append(CN_REPORT, block); append(TECH_REPORT, block)
    append(MASTER_LOG, f"\n[STAGE_DONE] {name} success={success} best={rec['best_predicted_log10_sigma']} gap={rec['gap_to_paper_minus_1p036']} next={next_decision}\n")

def stage(name, func, command):
    start = now(); log_path = LOGDIR / f"{len(STAGES):02d}_{name}.log"
    append(MASTER_LOG, f"\n========== {start} =========\n[STAGE_START] {name}\n[CMD] {command}\n")
    oldout, olderr = sys.stdout, sys.stderr
    tee = Tee(log_path, MASTER_LOG)
    sys.stdout = sys.stderr = tee
    result_files=[]; success=True; error=''; next_decision='continue'
    try:
        result_files, next_decision = func()
    except Exception as e:
        success=False; error=repr(e); traceback.print_exc(); next_decision='record failure and continue independent stages'
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        for f in tee.files: f.close()
    end = now(); record_stage(name,start,end,success,command,log_path,result_files,error,next_decision)

PRED = None
def predictor():
    global PRED
    if PRED is None: PRED = CommonPredictor()
    return PRED

def stage_env():
    out = RESULTS / '00_environment_snapshot.txt'
    cmds = [
        'hostname', 'pwd', 'git -C '+str(ML)+' status --short',
        'ls -lh '+str(ML/'results/final_metrics_comparison.csv'),
        'cat '+str(ML/'results/final_metrics_comparison.csv'),
        'cat '+str(ML/'results/ai_discovery_best_recipe.csv'),
        'find '+str(BASE/'optimizer_experiments/results')+' -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\\n" | sort',
        '/root/autodl-tmp/miniconda3/miniconda3/envs/matcond-repro/bin/python - <<PY2\nimport torch, sklearn, xgboost, scipy, sys\nprint(sys.executable)\nprint("torch", torch.__version__, torch.cuda.is_available())\nprint("sklearn", sklearn.__version__)\nprint("xgboost", xgboost.__version__)\nprint("scipy", scipy.__version__)\ntry:\n import optuna; print("optuna", optuna.__version__)\nexcept Exception as e: print("optuna FAIL", repr(e))\nPY2'
    ]
    text=[]
    for c in cmds:
        text.append('$ '+c)
        p=subprocess.run(c,shell=True,text=True,capture_output=True)
        text.append(p.stdout+p.stderr)
    out.write_text('\n'.join(text), encoding='utf-8')
    append(CN_REPORT, '# 自主逆向设计优化报告\n\n本报告由无人值守总控脚本持续追加。硬性限制：不运行 CHGNet MD，不运行 DFT/QE，不覆盖原结果。\n')
    append(TECH_REPORT, '# Autonomous Inverse Design Report\n\n')
    return [out], 'build common predictor and direct checks'

def direct_checks():
    p = predictor()
    items = [('paper_candidate','Sc','Mg',0.075,0.0319,1505,'论文候选'), ('current_ga','Sc','Mg',0.0627,0.0497,1524.9625579512503,'当前复现 GA rounded')]
    optsum = BASE/'optimizer_experiments/results/optimizer_comparison_summary.json'
    if optsum.exists():
        j=json.loads(optsum.read_text())
        b=j.get('best_by_predicted_log10_sigma') or {}
        if b: items.append(('optimizer_best', b['dopant_1'],b['dopant_2'],b['f1'],b['f2'],b['sintering_temperature'],'已有 optimizer best'))
    rows=[]
    for name,d1,d2,f1,f2,t,note in items:
        r=result_to_dict(p.predict_candidate(d1,d2,f1,f2,t,800)); r['candidate_name']=name; r['notes']=note; rows.append(r)
    update_best(rows); ALL_CANDIDATES.extend(rows)
    save_rows(RESULTS/'01_direct_candidate_check.csv', rows); save_rows(RESULTS/'direct_candidate_check.csv', rows)
    paper = next(r for r in rows if r['candidate_name']=='paper_candidate')
    diag = RESULTS/'diagnosis_prediction_surface.md'
    msg = 'checkpoint prediction surface differs from paper target' if abs(paper['predicted_log10_sigma']-PAPER_TARGET)>0.05 else 'paper candidate remains close to target; optimizer issue likely'
    diag.write_text(f"# Diagnosis\n\nPaper candidate prediction: {paper['predicted_log10_sigma']:.6f}\nTarget: {PAPER_TARGET}\nDecision: {msg}\n", encoding='utf-8')
    return [RESULTS/'01_direct_candidate_check.csv', RESULTS/'direct_candidate_check.csv', diag], msg

def candidate_dict(res, method, seed='', rank_note=''):
    d=result_to_dict(res); d['method']=method; d['seed']=seed; d['rank_note']=rank_note; return d

def eval_batch(cands, method, seed='', note=''):
    p=predictor(); out=[]; bs=4096
    for i in range(0,len(cands),bs):
        for r in p.predict_batch(cands[i:i+bs],800,source=method,notes=note): out.append(candidate_dict(r,method,seed,note))
    return out

def enhanced_ga_run():
    rng_seeds=[0,1,2,3,4]
    rows=[]; conv=[]
    dopants=list(DOPANTS)
    medium_needed=True
    for budget,g,pop in [('quick',25,60),('medium',100,200)]:
        budget_best=-999
        for seed in rng_seeds:
            rng=random.Random(seed)
            population=[]
            def make_ind():
                d1,d2=rng.sample(dopants,2); f1=rng.uniform(.04,.10); f2=rng.uniform(.01,.08); t=rng.uniform(1300,1600)
                if f1+f2<.08: f2=.08-f1+rng.uniform(0,.01)
                if f1+f2>.20: f2=max(.01,.20-f1-rng.uniform(0,.005))
                return [d1,d2,f1,f2,t]
            population=[make_ind() for _ in range(pop)]
            best=None; stagnant=0
            for gen in range(g):
                evals=eval_batch([(a,b,c,d,e) for a,b,c,d,e in population], 'enhanced_ga', str(seed), budget)
                evals_sorted=sorted(evals,key=lambda x:x['predicted_log10_sigma'], reverse=True)
                if best is None or evals_sorted[0]['predicted_log10_sigma']>best['predicted_log10_sigma']:
                    best=evals_sorted[0]; stagnant=0
                else: stagnant+=1
                conv.append({'budget':budget,'seed':seed,'generation':gen,'best':best['predicted_log10_sigma'],'diversity':len({(x[0],x[1],round(x[2],3),round(x[3],3)) for x in population})/len(population)})
                elites=[]
                for e in evals_sorted[:max(2,int(.08*pop))]: elites.append([e['dopant_1'],e['dopant_2'],e['f1_fraction'],e['f2_fraction'],e['sintering_temperature']])
                survivors=[]
                for _ in range(max(4,int(.35*pop))):
                    tour=rng.sample(evals_sorted, min(4,len(evals_sorted)))
                    w=max(tour,key=lambda x:x['predicted_log10_sigma']); survivors.append([w['dopant_1'],w['dopant_2'],w['f1_fraction'],w['f2_fraction'],w['sintering_temperature']])
                new=elites[:]
                mut_scale=max(.15,.55*(1-gen/max(1,g)))
                while len(new)<pop:
                    p1,p2=rng.choice(survivors),rng.choice(survivors)
                    child=[p1[0],p2[1] if rng.random()<.5 else p1[1], (p1[2]+p2[2])/2, (p1[3]+p2[3])/2, (p1[4]+p2[4])/2]
                    if child[0]==child[1]: child[1]=rng.choice([x for x in dopants if x!=child[0]])
                    if rng.random()<.35: child[2]*=rng.uniform(1-.25*mut_scale,1+.25*mut_scale)
                    if rng.random()<.35: child[3]*=rng.uniform(1-.25*mut_scale,1+.25*mut_scale)
                    if rng.random()<.20: child[4]+=rng.uniform(-80,80)*mut_scale
                    if rng.random()<.10: child[rng.choice([0,1])] = rng.choice(dopants)
                    child[2]=min(.10,max(.04,child[2])); child[3]=min(.08,max(.01,child[3])); child[4]=min(1600,max(1300,child[4]))
                    if child[0]==child[1]: child[1]=rng.choice([x for x in dopants if x!=child[0]])
                    if child[2]+child[3]<.08: child[3]=min(.08,.08-child[2]+.001)
                    if child[2]+child[3]>.20: child[3]=max(.01,.20-child[2]-.001)
                    new.append(child)
                population=new
                if stagnant>=50: break
            rows.append(best); budget_best=max(budget_best,best['predicted_log10_sigma'])
        if budget=='quick' and budget_best>FINE_GRID_BASELINE: break
    update_best(rows); ALL_CANDIDATES.extend(rows)
    save_rows(RESULTS/'enhanced_ga_top100.csv', sorted(rows,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:100]); save_rows(RESULTS/'enhanced_ga_convergence.csv', conv)
    write_json(RESULTS/'enhanced_ga_summary.json', {'best': best_from_rows(rows), 'n_results': len(rows)})
    decision='continue to systematic search; enhanced GA exceeded fine_grid baseline' if (best_from_rows(rows) or {}).get('predicted_log10_sigma',-999)>FINE_GRID_BASELINE else 'continue to medium/systematic search; no sufficient improvement'
    return [RESULTS/'enhanced_ga_top100.csv', RESULTS/'enhanced_ga_convergence.csv', RESULTS/'enhanced_ga_summary.json'], decision

def systematic_search():
    rows=[]; rng=np.random.default_rng(2026)
    # random 100k
    c=[]
    for _ in range(100000):
        d1,d2=rng.choice(DOPANTS,2,replace=False).tolist(); f1=float(rng.uniform(.04,.10)); f2=float(rng.uniform(.01,.08))
        if f1+f2<.08: f2=min(.08,.08-f1+rng.uniform(0,.02))
        if f1+f2>.20: f2=max(.01,.20-f1-rng.uniform(0,.01))
        c.append((d1,d2,f1,f2,float(rng.uniform(1300,1600))))
    rand=eval_batch(c,'random_search_100k','2026','100000 samples'); rows+=rand; save_rows(RESULTS/'search_random_top1000.csv', sorted(rand,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:1000])
    # grid all pair coarse/fine + high-res ScMg
    grid_c=[]
    for d1 in DOPANTS:
      for d2 in DOPANTS:
        if d1==d2: continue
        for f1 in np.arange(.04,.1001,.005):
          for f2 in np.arange(.01,.0801,.005):
            if .08<=f1+f2<=.20:
              for t in np.arange(1300,1600.1,50): grid_c.append((d1,d2,float(f1),float(f2),float(t)))
    grid=eval_batch(grid_c,'grid_coarse','NA','coarse all-pair');
    top_pairs=[]
    for r in sorted(grid,key=lambda x:x['predicted_log10_sigma'],reverse=True):
        pair=(r['dopant_1'],r['dopant_2'])
        if pair not in top_pairs: top_pairs.append(pair)
        if len(top_pairs)>=10: break
    fine_c=[]
    for d1,d2 in top_pairs+[('Sc','Mg'),('Mg','Sc')]:
        for f1 in np.arange(.05,.1001,.001):
          for f2 in np.arange(.01,.0801,.001):
            if .08<=f1+f2<=.20:
              for t in np.arange(1400,1600.1,10): fine_c.append((d1,d2,float(f1),float(f2),float(t)))
    fine=eval_batch(fine_c,'fine_grid','NA','top-pair fine grid')
    rows += grid + fine
    save_rows(RESULTS/'search_grid_top1000.csv', sorted(grid+fine,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:1000])
    save_rows(RESULTS/'search_algorithm_comparison.csv', [best_from_rows([r for r in rows if r['method']==m]) for m in sorted(set(r['method'] for r in rows)) if best_from_rows([r for r in rows if r['method']==m])])
    update_best(rows); ALL_CANDIDATES.extend(rows)
    # Optuna if installed or install
    opt_rows=[]
    try:
      import optuna
    except Exception:
      subprocess.run([sys.executable,'-m','pip','install','optuna'], check=False)
      try: import optuna
      except Exception: optuna=None
    if optuna:
      for seed in [0,1,2,3,4]:
        sampler=optuna.samplers.TPESampler(seed=seed); study=optuna.create_study(direction='maximize', sampler=sampler)
        def obj(trial):
          d1=trial.suggest_categorical('d1',DOPANTS); d2=trial.suggest_categorical('d2',DOPANTS)
          if d1==d2: return -999
          f1=trial.suggest_float('f1',.04,.10); f2=trial.suggest_float('f2',.01,.08)
          if not (.08<=f1+f2<=.20): return -999
          t=trial.suggest_float('temp',1300,1600)
          rr=eval_batch([(d1,d2,f1,f2,t)],'optuna_tpe',str(seed),'10000 trials')[0]
          trial.set_user_attr('row',rr); return rr['predicted_log10_sigma']
        study.optimize(obj, n_trials=10000, show_progress_bar=False)
        for tr in sorted([t for t in study.trials if t.value is not None], key=lambda t:t.value, reverse=True)[:100]:
          if 'row' in tr.user_attrs: opt_rows.append(tr.user_attrs['row'])
    save_rows(RESULTS/'search_optuna_top1000.csv', sorted(opt_rows,key=lambda x:x.get('predicted_log10_sigma',-999),reverse=True)[:1000])
    rows += opt_rows
    # DE/SA lightweight
    try:
      from scipy.optimize import differential_evolution
      de=[]
      pairs=[]
      for r in sorted(rows,key=lambda x:x['predicted_log10_sigma'],reverse=True):
        pair=(r['dopant_1'],r['dopant_2'])
        if pair not in pairs: pairs.append(pair)
        if len(pairs)>=10: break
      for d1,d2 in pairs:
        def obj(x):
          f1,f2,t=x
          if not(.08<=f1+f2<=.20): return 999
          return -eval_batch([(d1,d2,float(f1),float(f2),float(t))],'differential_evolution','2026','top10 pairs')[0]['predicted_log10_sigma']
        res=differential_evolution(obj,[(.04,.10),(.01,.08),(1300,1600)],seed=2026,maxiter=35,popsize=8,polish=True)
        de.append(eval_batch([(d1,d2,float(res.x[0]),float(res.x[1]),float(res.x[2]))],'differential_evolution','2026','top10 pairs')[0])
      save_rows(RESULTS/'search_de_top1000.csv', sorted(de,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:1000]); rows+=de
    except Exception as e:
      (RESULTS/'search_de_top1000.csv').write_text('error\n'+repr(e)+'\n')
    # SA from top
    sa=[]
    for base in sorted(rows,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:20]:
      cur=base.copy(); temp=0.05; rng=random.Random(777)
      for step in range(300):
        d1,d2=cur['dopant_1'],cur['dopant_2']
        if rng.random()<.05: d1=rng.choice(DOPANTS)
        if rng.random()<.05: d2=rng.choice([d for d in DOPANTS if d!=d1])
        f1=min(.10,max(.04,cur['f1_fraction']+rng.uniform(-.005,.005)))
        f2=min(.08,max(.01,cur['f2_fraction']+rng.uniform(-.005,.005)))
        if not(.08<=f1+f2<=.20): continue
        t=min(1600,max(1300,cur['sintering_temperature']+rng.uniform(-20,20)))
        nr=eval_batch([(d1,d2,f1,f2,t)],'simulated_annealing','777','local jump')[0]
        if nr['predicted_log10_sigma']>cur['predicted_log10_sigma'] or rng.random()<math.exp((nr['predicted_log10_sigma']-cur['predicted_log10_sigma'])/max(temp,1e-6)):
          cur=nr
        temp*=.995
      sa.append(cur)
    save_rows(RESULTS/'search_sa_top1000.csv', sorted(sa,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:1000]); rows+=sa
    update_best(rows); ALL_CANDIDATES.extend(rows)
    # pareto simple
    candidates=sorted(rows,key=lambda x:x['predicted_log10_sigma'],reverse=True)[:5000]
    pareto=[]
    for r in candidates:
      dominated=False
      for q in candidates[:1000]:
        if q is r: continue
        if q['predicted_log10_sigma']>=r['predicted_log10_sigma'] and q['radius_mismatch_pm']<=r['radius_mismatch_pm'] and q['distance_to_training_scaled']<=r['distance_to_training_scaled'] and (q['predicted_log10_sigma']>r['predicted_log10_sigma'] or q['radius_mismatch_pm']<r['radius_mismatch_pm']): dominated=True; break
      if not dominated: pareto.append(r)
      if len(pareto)>=1000: break
    save_rows(RESULTS/'search_pareto_front.csv', pareto)
    return [RESULTS/'search_random_top1000.csv',RESULTS/'search_grid_top1000.csv',RESULTS/'search_optuna_top1000.csv',RESULTS/'search_de_top1000.csv',RESULTS/'search_sa_top1000.csv',RESULTS/'search_pareto_front.csv',RESULTS/'search_algorithm_comparison.csv'], 'merge candidates and run checkpoint diagnosis if needed'

def checkpoint_diagnosis():
    p = predictor(); direct = pd.read_csv(RESULTS/'direct_candidate_check.csv').to_dict('records')
    paper = next((r for r in direct if r.get('candidate_name')=='paper_candidate'), None)
    ckpts = sorted(set([str(x) for x in (BASE).glob('**/*.pth') if 'autonomous_inverse_design/model_ensembles' not in str(x)]))
    rows=[]
    targets=[('paper','Sc','Mg',.075,.0319,1505),('current_ga','Sc','Mg',.0627,.0497,1524.9625579512503)]
    if BEST: targets.append(('current_best',BEST['dopant_1'],BEST['dopant_2'],BEST['f1_fraction'],BEST['f2_fraction'],BEST['sintering_temperature']))
    for ck in ckpts:
      try:
        pp=CommonPredictor(checkpoint_path=ck)
        for name,d1,d2,f1,f2,t in targets:
          r=result_to_dict(pp.predict_candidate(d1,d2,f1,f2,t,800)); r['checkpoint']=ck; r['candidate_name']=name; rows.append(r)
      except Exception as e:
        rows.append({'checkpoint':ck,'candidate_name':'ERROR','error':repr(e)})
    save_rows(RESULTS/'checkpoint_candidate_matrix.csv', rows)
    md=RESULTS/'checkpoint_diagnosis_report.md'
    md.write_text('# Checkpoint diagnosis\n\n' + f'Found {len(ckpts)} pth checkpoints. Paper candidate current prediction: {paper.get("predicted_log10_sigma") if paper else None}\n', encoding='utf-8')
    need_multiseed = bool(paper and abs(paper['predicted_log10_sigma']-PAPER_TARGET)>0.05)
    return [RESULTS/'checkpoint_candidate_matrix.csv', md], 'current checkpoint differs from paper target; multiseed retraining is allowed but deferred unless separately needed' if need_multiseed else 'paper target supported; skip multiseed retraining'

def merge_and_rank():
    rows=list(ALL_CANDIDATES)
    for f in ['search_random_top1000.csv','search_grid_top1000.csv','search_optuna_top1000.csv','search_de_top1000.csv','search_sa_top1000.csv','enhanced_ga_top100.csv','direct_candidate_check.csv']:
      p=RESULTS/f
      if p.exists() and p.stat().st_size>0:
        try: rows += pd.read_csv(p).to_dict('records')
        except Exception: pass
    # normalize dedup
    seen=set(); uniq=[]
    for r in rows:
      if 'predicted_log10_sigma' not in r: continue
      key=(r.get('dopant_1'),r.get('dopant_2'),round(float(r.get('f1_fraction',0)),5),round(float(r.get('f2_fraction',0)),5),round(float(r.get('sintering_temperature',0)),2))
      if key not in seen: seen.add(key); uniq.append(r)
    save_rows(RESULTS/'all_candidates_merged.csv', uniq)
    score=sorted(uniq,key=lambda x:float(x.get('predicted_log10_sigma',-999)),reverse=True)
    save_rows(RESULTS/'ranking_score_only.csv', score[:1000])
    stab=[r for r in score if r.get('plausible_region') in ['stable_like_ysz','metastable_plausible'] and float(r.get('distance_to_training_scaled',0))<=0.25]
    save_rows(RESULTS/'ranking_stability_filtered.csv', stab[:1000])
    scmg=[r for r in score if {r.get('dopant_1'),r.get('dopant_2')}=={'Sc','Mg'}]
    save_rows(RESULTS/'ranking_sc_mg_only.csv', scmg[:1000])
    robust=[r for r in stab if float(r.get('radius_mismatch_pm',99))<9.0]
    save_rows(RESULTS/'ranking_ensemble_robust.csv', robust[:1000])
    feasible=[]
    for r in stab[:2000]:
      n1=round(float(r.get('f1_fraction',0))*108); n2=round(float(r.get('f2_fraction',0))*108)
      if n1+n2>0 and n1+n2<25: feasible.append(r)
    save_rows(RESULTS/'ranking_dft_feasible.csv', feasible[:1000])
    global BEST; BEST=score[0] if score else BEST
    rec=[]
    labels=[('best_by_model_score', score[0] if score else None),('best_physically_plausible', stab[0] if stab else None),('best_sc_mg_candidate', scmg[0] if scmg else None),('closest_to_paper_candidate', min(uniq,key=lambda r: (r.get('dopant_1')!='Sc' or r.get('dopant_2')!='Mg', abs(float(r.get('f1_fraction',0))-.075)+abs(float(r.get('f2_fraction',0))-.0319)+abs(float(r.get('sintering_temperature',0))-1505)/1000)) if uniq else None),('recommended_for_chgnet', stab[0] if stab else (score[0] if score else None)),('recommended_for_dft', feasible[0] if feasible else (stab[0] if stab else None))]
    for label,r in labels:
      if r: rr=dict(r); rr['recommendation_role']=label; rec.append(rr)
    save_rows(RESULTS/'final_recommended_candidates.csv', rec)
    summary={'completed':True,'best':BEST,'recommendations':rec,'exceeds_current_ga': BEST and float(BEST['predicted_log10_sigma'])>CURRENT_GA_BASELINE,'exceeds_fine_grid': BEST and float(BEST['predicted_log10_sigma'])>FINE_GRID_BASELINE,'reaches_paper_target': BEST and float(BEST['predicted_log10_sigma'])>=PAPER_TARGET,'no_chgnet_md':True,'no_dft_qe':True}
    write_json(RESULTS/'final_summary.json', summary)
    return [RESULTS/'all_candidates_merged.csv',RESULTS/'ranking_score_only.csv',RESULTS/'ranking_stability_filtered.csv',RESULTS/'ranking_sc_mg_only.csv',RESULTS/'ranking_ensemble_robust.csv',RESULTS/'ranking_dft_feasible.csv',RESULTS/'final_recommended_candidates.csv',RESULTS/'final_summary.json'], 'write final reports'

def final_reports():
    summary=json.loads((RESULTS/'final_summary.json').read_text())
    b=summary.get('best') or {}
    rec=summary.get('recommendations') or []
    def line(r): return f"{r.get('recommendation_role','')} | {r.get('dopant_1')}-{r.get('dopant_2')} | {float(r.get('f1_mol_percent',0)):.2f}%/{float(r.get('f2_mol_percent',0)):.2f}% | log10σ={float(r.get('predicted_log10_sigma',-999)):.4f} | {r.get('supercell_108_suggestion','')}"
    append(CN_REPORT, f"\n# 最终结论\n\n- 最佳算法/来源：{b.get('method','unknown')}\n- 最佳候选：{b.get('dopant_1')}-{b.get('dopant_2')}，{b.get('f1_mol_percent')} / {b.get('f2_mol_percent')} mol%\n- predicted_log10_sigma：{b.get('predicted_log10_sigma')}\n- 是否超过当前 GA：{summary.get('exceeds_current_ga')}\n- 是否超过已有 fine_grid -1.3399：{summary.get('exceeds_fine_grid')}\n- 是否达到论文 -1.036：{summary.get('reaches_paper_target')}\n- 是否建议替换论文主候选：不建议直接替换；建议作为当前 checkpoint 下新增候选，并保留论文 Sc-Mg baseline。\n- 是否建议修改 CHGNet/DFT 配比：建议后续新增当前最优和稳定筛选候选，但本阶段不运行。\n\n## 推荐候选\n\n" + '\n'.join('- '+line(r) for r in rec) + "\n\n## 可靠性说明\n\n这些结论只是在当前 PIML checkpoint 和当前特征构造下的 ML screening 假设；CHGNet/DFT 未运行，因此不能作为完成验证。\n")
    append(TECH_REPORT, (RESULTS/'final_summary.json').read_text()+"\n")
    return [CN_REPORT, TECH_REPORT], 'autonomous inverse design completed'

def main():
    sys.stdout = sys.stderr = Tee(MASTER_LOG)
    print(f"========== autonomous inverse design main started {now()} ==========")
    stages=[('00_environment',stage_env,'environment snapshot'),('01_direct_predictor',direct_checks,'direct candidate checks'),('02_enhanced_ga',enhanced_ga_run,'enhanced GA quick/medium'),('03_systematic_search',systematic_search,'random/grid/optuna/de/sa/pareto'),('04_checkpoint_diagnosis',checkpoint_diagnosis,'checkpoint diagnosis'),('05_merge_rank',merge_and_rank,'merge and rank candidates'),('06_final_reports',final_reports,'final reports')]
    start=time.time()
    for name,fn,cmd in stages:
      if time.time()-start>24*3600: break
      stage(name,fn,cmd)
      if BEST and BEST.get('predicted_log10_sigma',-999)>=STOP_TARGET:
        append(MASTER_LOG, f"[STOP_RULE] reached paper target with {BEST.get('predicted_log10_sigma')}\n")
        # still continue to merge/final if not done
    print(f"========== autonomous inverse design main ended {now()} ==========")

if __name__ == '__main__': main()
