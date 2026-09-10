"""Train 내부 시간 CV. python -m frauddetectionreport.model_validation"""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import features, splits, evaluate, policy, config
from .costs import DEFAULT_PARAMS


def run():
    output = config.PROJECT_ROOT / 'reports' / 'model_validation'
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(config.TRAIN_PARQUET)
    raw = raw.loc[splits.make_time_split(raw).train].reset_index(drop=True)
    folds = splits.time_series_folds(raw, np.ones(len(raw), dtype=bool), n_folds=4)
    params = dict(objective='binary', learning_rate=.03, num_leaves=127,
                  min_child_samples=80, feature_fraction=.6, bagging_fraction=.8,
                  bagging_freq=1, lambda_l2=1., verbosity=-1, seed=42,
                  num_threads=4, deterministic=True, force_col_wise=True)
    rounds = 300
    (output / 'protocol.json').write_text(json.dumps(dict(params=params, rounds=rounds,
        selection='mean fold raw average precision; no calibration or test selection',
        note='Follow-up experiment after prior test inspection; not a fresh holdout.'), indent=2), encoding='utf-8')
    rows = []
    for i, (tr, va) in enumerate(folds):
        d = raw.loc[tr | va].copy()
        mask = tr[tr | va]
        print(f'fold {i}: building features', flush=True)
        f, _ = features.build_features(d, mask)
        # Restrict categorical vocabularies to this fold's learning data.
        for col in f.select_dtypes(include='category').columns:
            cats = f.loc[mask, col].dropna().unique()
            f[col] = f[col].cat.set_categories(cats)
        y = f.isFraud.to_numpy()
        weight = (y[mask] == 0).sum() / (y[mask] == 1).sum()
        v = f.loc[~mask]
        pred = v[['TransactionID', 'TransactionDT', 'isFraud', 'TransactionAmt']].copy()
        for weighted in [False, True]:
            for uid in [False, True]:
                name = f'{"weighted" if weighted else "plain"}_{"uid" if uid else "no_uid"}'
                print(f'fold {i}: {name}, {rounds} rounds', flush=True)
                cols = [c for c in features.model_columns(f) if uid or not c.startswith('uid')]
                model = lgb.train(dict(params, scale_pos_weight=float(weight) if weighted else 1.),
                    lgb.Dataset(f.loc[mask, cols], label=y[mask]), num_boost_round=rounds)
                prob = model.predict(f.loc[~mask, cols])
                pred[name] = prob
                r = evaluate.model_report(y[~mask], prob)
                cost = evaluate.evaluate_policy(policy.AmountAwareThreshold(), prob,
                    v.TransactionAmt.to_numpy(), y[~mask], DEFAULT_PARAMS)['total']
                row = dict(fold=i, model=name, rounds=rounds, n_train=int(mask.sum()), n_valid=int((~mask).sum()),
                    train_max=int(f.loc[mask, 'TransactionDT'].max()), valid_min=int(v.TransactionDT.min()),
                    valid_max=int(v.TransactionDT.max()), p4_cost=cost, **r)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / 'fold_metrics.csv', index=False)
                print(f'  AP={r["pr_auc"]:.6f}, P4={cost:.2f}', flush=True)
                del model
                gc.collect()
        pred.to_parquet(output / f'fold_{i}_raw_predictions.parquet', index=False)
        del f, d, pred
        gc.collect()
    result = pd.DataFrame(rows)
    summary = result.groupby('model')[['pr_auc', 'roc_auc', 'brier', 'ece', 'p4_cost']].mean().sort_values('pr_auc', ascending=False)
    summary.to_csv(output / 'summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    run()
