#!/usr/bin/env python3
"""Create stratified samples from datasets and check coverage in raw inference files.

Usage: python scripts/evaluation/create_and_check_samples.py
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def stratified_sample(df, n=200, strat_col='page_views', bins=5, random_state=0):
    # create quantile bins on strat_col
    if strat_col not in df.columns:
        # fallback: simple random sample
        return df.sample(n=min(n, len(df)), random_state=random_state)
    try:
        df['__bin'] = pd.qcut(df[strat_col].fillna(0).astype(float), q=bins, duplicates='drop')
    except Exception:
        df['__bin'] = pd.cut(df[strat_col].fillna(0).astype(float), bins)
    groups = df.groupby('__bin')
    total = len(df)
    picks = []
    for name, g in groups:
        k = int(round(n * len(g) / total))
        if k <= 0:
            continue
        picks.append(g.sample(n=min(k, len(g)), random_state=random_state))
    if picks:
        sampled = pd.concat(picks)
    else:
        sampled = df.sample(n=min(n, len(df)), random_state=random_state)
    # adjust if we oversampled or undersampled
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)
    elif len(sampled) < n:
        remaining = df.drop(sampled.index)
        need = n - len(sampled)
        if len(remaining) > 0:
            sampled = pd.concat([sampled, remaining.sample(n=min(need, len(remaining)), random_state=random_state)])
    sampled = sampled.drop(columns=['__bin'], errors='ignore')
    return sampled


def load_questions_from_jsonl(path):
    qs = set()
    if not path.exists():
        return qs
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # skip metadata
            if isinstance(obj, dict) and obj.get('type') == 'metadata':
                continue
            q = obj.get('question') or obj.get('prompt') or obj.get('input')
            if q:
                qs.add(q.strip())
    return qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--bins', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='results/samples')
    parser.add_argument('--models', nargs='+', default=['Llama-3.1-8B-Instruct','DeepSeek-v3.2'])
    args = parser.parse_args()

    base = Path('.')
    proc = base / 'data' 
    out = base / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    subsets = [
        ('long_tailed.csv','long_tail'),
        ('anchor_induced.csv','mixed_fact'),
        ('near_true.csv','near_true'),
    ]

    report = {}
    for fname, subset_key in subsets:
        path = proc / fname
        if not path.exists():
            print(f"Missing dataset file: {path}")
            continue
        df = pd.read_csv(path)
        sampled = stratified_sample(df, n=args.n, strat_col='page_views', bins=args.bins)
        sample_path = out / f"sample_{subset_key}_{args.n}.csv"
        sampled.to_csv(sample_path, index=False)
        print(f"Wrote sample {sample_path} ({len(sampled)} rows)")

        # detect question column
        possible_qcols = ['question', 'grounded_question', 'ungrounded_question', 'prompt', 'input']
        qcol = None
        for c in possible_qcols:
            if c in sampled.columns:
                qcol = c
                break
        if qcol is None:
            # fallback to first string column
            for c in sampled.columns:
                if sampled[c].dtype == object:
                    qcol = c
                    break
        if qcol is None:
            print(f"No question-like column found in {path}; skipping coverage check")
            report[subset_key] = {'sample_csv': str(sample_path), 'n_sample': len(sampled), 'missing': {}}
            continue
        questions = set(sampled[qcol].astype(str).str.strip().tolist())
        report[subset_key] = {'sample_csv': str(sample_path), 'n_sample': len(sampled), 'missing': {}}

        # check each model's raw inference file for coverage
        for model in args.models:
            # look for files under results/inference_results/**/raw_*_{model}.jsonl
            pattern = base.glob(f"results/inference_results/**/raw_*_{model}.jsonl")
            found = list(pattern)
            if not found:
                report[subset_key]['missing'][model] = {'found_file': None, 'covered': 0, 'missing_questions': list(questions)}
                continue
            # choose the first matching file (should be the subset-specific one)
            # prefer file containing subset_key in path
            chosen = None
            for p in found:
                if subset_key in str(p):
                    chosen = p
                    break
            if chosen is None:
                chosen = found[0]
            available = load_questions_from_jsonl(chosen)
            missing = sorted([q for q in questions if q not in available])
            report[subset_key]['missing'][model] = {'found_file': str(chosen), 'covered': len(questions)-len(missing), 'missing_count': len(missing), 'missing_questions_sample': missing[:10]}

    out_report = out / 'coverage_report.json'
    with out_report.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Wrote coverage report to {out_report}")


if __name__ == '__main__':
    main()
