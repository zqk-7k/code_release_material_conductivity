#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re, subprocess, sys, hashlib
from pathlib import Path
from datetime import datetime

BASE = Path('/root/autodl-tmp/qkzhang/material-conductivity-reproduce')
AID = BASE / 'autonomous_inverse_design'
RESULTS = AID / 'results'
LOGDIR = BASE / 'logs' / 'autonomous_inverse_design'
CN = BASE / 'checkpoint回溯报告_中文.md'
MATRIX = RESULTS / 'checkpoint_backtracking_matrix.csv'
SEARCH_OUT = RESULTS / 'checkpoint_backtracking_search_hits.txt'
RESULTS.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AID / 'scripts'))
from common_predictor import CommonPredictor, result_to_dict

TARGET = -1.036
CANDIDATES = [
    ('paper_candidate_Sc7p50_Mg3p19', 'Sc', 'Mg', 0.0750, 0.0319, 1505.0, '论文候选'),
    ('current_optimizer_Mg4p00_Sc4p00', 'Mg', 'Sc', 0.0400, 0.0400, 1600.0, '当前优化候选；温度按优化边界上限 1600 C'),
    ('current_ga_Sc6p27_Mg4p97', 'Sc', 'Mg', 0.0627, 0.0497, 1524.9625579512503, '当前复现 GA 候选'),
]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def shell(cmd: str, cwd: Path = BASE):
    p=subprocess.run(cmd, shell=True, cwd=str(cwd), text=True, capture_output=True)
    return p.returncode, p.stdout + p.stderr

def find_files():
    ckpts = []
    for p in sorted(BASE.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.pth','.pt'} and 'baseline' not in str(p).lower():
            ckpts.append(p)
    csvs = sorted(BASE.rglob('ai_discovery_best_recipe.csv')) + sorted(BASE.rglob('virtual_screening_results.csv'))
    optim = []
    for p in BASE.rglob('*'):
        if p.is_file() and re.search(r'(optimizer|ga|history|convergence)', p.name, re.I):
            optim.append(p)
    backups = sorted([p for p in BASE.glob('backup_before_full_reproduce*') if p.is_dir()])
    return ckpts, csvs, sorted(set(optim)), backups

def search_text_hits():
    patterns = ['Sc', 'Mg', '7.50', '7.5', '3.19', '-1.036', '-1.04']
    exts = {'.csv','.txt','.json','.md'}
    hits=[]
    for p in BASE.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if p.stat().st_size > 10_000_000:
            continue
        try:
            text=p.read_text(errors='ignore')
        except Exception:
            continue
        if any(x in text for x in patterns):
            lines=[]
            for i,line in enumerate(text.splitlines(),1):
                if any(x in line for x in patterns):
                    lines.append(f'{i}: {line[:300]}')
                if len(lines)>=25: break
            hits.append((p, lines))
    with SEARCH_OUT.open('w', encoding='utf-8') as f:
        for p, lines in hits:
            f.write(f'===== {p} =====\n')
            f.write('\n'.join(lines)+'\n')
    return hits

def git_history_search():
    repo = BASE / 'material-conductivity-data-analysis-ml'
    out=[]
    if (repo/'.git').exists():
        cmds = [
            "git log --all --stat -- results/ai_discovery_best_recipe.csv results/final_metrics_comparison.csv results/checkpoint/piml/best_piml_model.pth",
            "git log --all -S'-1.036' -- .",
            "git log --all -S'7.50' -- .",
            "git log --all -S'3.19' -- .",
            "git grep -n -E '7\\.50|3\\.19|-1\\.036|-1\\.04|Sc.*Mg|Mg.*Sc' $(git rev-list --all) -- '*.csv' '*.txt' '*.json' '*.md' 2>/dev/null | head -200",
        ]
        for cmd in cmds:
            rc, txt = shell(cmd, repo)
            out.append(f'$ {cmd}\nrc={rc}\n{txt}\n')
    git_out = RESULTS / 'git_history_search.txt'
    git_out.write_text('\n'.join(out), encoding='utf-8')
    return git_out

def evaluate_checkpoints(ckpts):
    rows=[]
    for ck in ckpts:
        try:
            pred = CommonPredictor(checkpoint_path=str(ck))
            ck_sha = sha256(ck)
            for cname,d1,d2,f1,f2,temp,note in CANDIDATES:
                r = result_to_dict(pred.predict_candidate(d1,d2,f1,f2,temp,800))
                r.update({
                    'candidate_name': cname,
                    'checkpoint_path': str(ck),
                    'checkpoint_sha256': ck_sha,
                    'checkpoint_size': ck.stat().st_size,
                    'checkpoint_mtime': datetime.fromtimestamp(ck.stat().st_mtime).isoformat(),
                    'target_gap_vs_minus_1p036': r['predicted_log10_sigma'] - TARGET,
                    'supports_paper_target_tol_0p05': abs(r['predicted_log10_sigma'] - TARGET) <= 0.05,
                    'backtracking_notes': note,
                })
                rows.append(r)
        except Exception as e:
            rows.append({'checkpoint_path': str(ck), 'candidate_name': 'ERROR', 'error': repr(e)})
    if rows:
        keys=sorted({k for r in rows for k in r})
        with MATRIX.open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows

def summarize(rows, ckpts, csvs, optim, backups, hits, git_out):
    paper_rows=[r for r in rows if r.get('candidate_name')=='paper_candidate_Sc7p50_Mg3p19' and 'predicted_log10_sigma' in r]
    support=[r for r in paper_rows if r.get('supports_paper_target_tol_0p05')]
    best_paper=max(paper_rows, key=lambda r:r['predicted_log10_sigma']) if paper_rows else None
    csv_hits=[]
    for p in csvs:
        try:
            txt=p.read_text(errors='ignore')
            if any(s in txt for s in ['7.50','7.5','3.19','-1.036','-1.04']):
                csv_hits.append(str(p))
        except Exception: pass
    lines=[]
    lines.append('# checkpoint 回溯报告')
    lines.append('')
    lines.append(f'生成时间：{datetime.now().isoformat(timespec="seconds")}')
    lines.append('')
    lines.append('## 结论')
    lines.append('')
    if support:
        lines.append(f'- 找到能支持论文候选 -1.036±0.05 的 checkpoint：{support[0]["checkpoint_path"]}，预测值 {support[0]["predicted_log10_sigma"]:.4f}。')
    else:
        lines.append('- 未找到能支持论文候选 `Sc=7.50 mol%, Mg=3.19 mol%, predicted log10σ≈-1.036` 的 checkpoint。')
        if best_paper:
            lines.append(f'- 所有 checkpoint 中论文候选最高预测值为 {best_paper["predicted_log10_sigma"]:.4f}，checkpoint: `{best_paper["checkpoint_path"]}`，与 -1.036 差 {best_paper["target_gap_vs_minus_1p036"]:+.4f}。')
    lines.append('- 本阶段没有重新训练，没有运行 CHGNet MD，没有运行 DFT/QE。')
    lines.append('')
    lines.append('## 搜索范围')
    lines.append('')
    lines.append(f'- checkpoint 数量：{len(ckpts)}')
    lines.append(f'- ai_discovery/virtual_screening CSV 数量：{len(csvs)}')
    lines.append(f'- optimizer/GA/history 文件数量：{len(optim)}')
    lines.append(f'- backup_before_full_reproduce 目录数量：{len(backups)}')
    lines.append(f'- 文本命中数量：{len(hits)}，详见 `{SEARCH_OUT}`')
    lines.append(f'- git 历史搜索输出：`{git_out}`')
    lines.append('')
    lines.append('## checkpoint 评估摘要')
    lines.append('')
    lines.append('| checkpoint | paper Sc7.50/Mg3.19 | Mg4/Sc4 | Sc6.27/Mg4.97 | 是否支持论文 -1.036 |')
    lines.append('|---|---:|---:|---:|---|')
    by_ck={}
    for r in rows:
        if 'predicted_log10_sigma' not in r: continue
        by_ck.setdefault(r['checkpoint_path'], {})[r['candidate_name']] = r
    for ck, vals in by_ck.items():
        p=vals.get('paper_candidate_Sc7p50_Mg3p19', {})
        o=vals.get('current_optimizer_Mg4p00_Sc4p00', {})
        g=vals.get('current_ga_Sc6p27_Mg4p97', {})
        lines.append(f"| `{ck}` | {p.get('predicted_log10_sigma','')} | {o.get('predicted_log10_sigma','')} | {g.get('predicted_log10_sigma','')} | {p.get('supports_paper_target_tol_0p05','')} |")
    lines.append('')
    lines.append('## 旧 CSV / 文本证据')
    lines.append('')
    if csv_hits:
        lines.append('以下 CSV 包含论文候选相关关键词：')
        for p in csv_hits: lines.append(f'- `{p}`')
    else:
        lines.append('- 未在 ai_discovery/virtual_screening CSV 中发现 `7.50/3.19/-1.036/-1.04` 这类旧论文候选数值。')
    lines.append('')
    lines.append('## 最可能原因')
    lines.append('')
    if not support:
        lines.append('- 当前仓库中可见的 checkpoint 都不能让论文候选达到 -1.036，说明问题更可能来自未保存的旧 checkpoint、旧训练数据快照、旧特征构造/预测逻辑、或论文中使用的历史模型与当前仓库 checkpoint 不一致。')
        lines.append('- 备份目录里的 checkpoint 与当前 checkpoint 很可能是同一轮或相近文件，不能恢复论文候选预测面。')
    lines.append('')
    lines.append('## 论文候选处理建议')
    lines.append('')
    lines.append('- 不建议直接删除论文原候选；它仍应作为论文/历史结果的 baseline 或目标候选保留。')
    lines.append('- 在当前 checkpoint 下，建议新增说明：当前可复现模型预测面下最优候选是 Mg-Sc 4.0/4.0 左右，而论文候选在当前 checkpoint 下无法复现到 -1.036。')
    lines.append('- 如果论文需要严格复现原 GA 候选，下一步应寻找原始训练 checkpoint、原始数据快照或原始特征工程版本，而不是继续调优化器。')
    lines.append('')
    lines.append('## 输出文件')
    lines.append('')
    lines.append(f'- matrix: `{MATRIX}`')
    lines.append(f'- text search hits: `{SEARCH_OUT}`')
    lines.append(f'- git search: `{git_out}`')
    CN.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    summary={
        'found_supporting_checkpoint': bool(support),
        'supporting_checkpoint': support[0]['checkpoint_path'] if support else None,
        'best_paper_prediction': best_paper,
        'matrix': str(MATRIX),
        'report': str(CN),
        'no_training': True,
        'no_chgnet_md': True,
        'no_dft_qe': True,
    }
    (RESULTS/'checkpoint_backtracking_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def main():
    ckpts,csvs,optim,backups=find_files()
    hits=search_text_hits()
    git_out=git_history_search()
    rows=evaluate_checkpoints(ckpts)
    summary=summarize(rows,ckpts,csvs,optim,backups,hits,git_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
