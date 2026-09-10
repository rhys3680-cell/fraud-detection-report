"""06의 추가 운영 가정 실험. 기존 모델 확률만 사용한다."""
import numpy as np
import pandas as pd

from .costs import Action, DEFAULT_PARAMS, expected_cost, optimal_action
from . import evaluate, policy


class ArrivalReview:
    """시간순 입력에서 양의 검토 절감액 후보를 선착순으로 할당한다.

    미래 후보의 점수나 레이블을 사용하지 않는 단순 온라인 비교 기준이다.
    decide 호출 하나는 완전한 일별 입력을 포함해야 한다 (호출 간 상태 없음).
    """
    def __init__(self, with_auth=False):
        self.with_auth = with_auth
        self.name = "P6_arrival" if with_auth else "P5_arrival"

    def decide(self, prob, amount, p, review_group=None):
        allowed = (Action.APPROVE, Action.AUTH, Action.BLOCK) if self.with_auth else (Action.APPROVE, Action.BLOCK)
        base = optimal_action(prob, amount, p, allowed=allowed)
        saving = expected_cost(base, prob, amount, p) - expected_cost(Action.REVIEW, prob, amount, p)
        groups = np.zeros(len(prob)) if review_group is None else np.asarray(review_group)
        if len(groups) != len(prob):
            raise ValueError("review_group length mismatch")
        used = {}
        for i, group in enumerate(groups):
            if saving[i] > 0 and (p.review_budget is None or used.get(group, 0) < p.review_budget):
                base[i] = Action.REVIEW
                used[group] = used.get(group, 0) + 1
        return base


def run(pred):
    """선택·확인 기간을 동일한 고정 그리드로 진단한다 (후속 분석)."""
    sel = pred[pred.half == "a_select"]
    budget = max(1, int((sel.TransactionDT // 86400).value_counts().median() * .01))
    params = DEFAULT_PARAMS.with_(review_budget=budget)
    grids = {
        "auth_friction": [0, .5, 1, 2, 5, 10, 20, 50, 100],
        "review_catch_rate": [0, .25, .5, .7, .8, .9, .95, 1],
        "review_delay_cost": [0, 1, 3, 5, 10, 25, 50],
    }
    sensitivity, allocation = [], []
    for half in ["a_select", "b_confirm"]:
        d = pred[pred.half == half].sort_values(["TransactionDT", "TransactionID"], kind="stable")
        prob, amount, y = d.p.to_numpy(), d.amount.to_numpy(), d.is_fraud.to_numpy()
        days = (d.TransactionDT // 86400).to_numpy()
        for key, values in grids.items():
            for value in values:
                for pol in [policy.AmountAwareThreshold(), policy.WithReview(), policy.FullPolicy()]:
                    r = evaluate.evaluate_policy(pol, prob, amount, y, params.with_(**{key: value}), review_group=days)
                    sensitivity.append(dict(half=half, parameter=key, value=value, **r))
        for pol in [policy.AmountAwareThreshold(), policy.WithReview(), ArrivalReview(), policy.FullPolicy(), ArrivalReview(True)]:
            r = evaluate.evaluate_policy(pol, prob, amount, y, params, review_group=days)
            allocation.append(dict(half=half, **r))
    return pd.DataFrame(sensitivity), pd.DataFrame(allocation)


def paired_block_bootstrap(daily_differences, block_days=3, n_resamples=10000, seed=42):
    """일별 비용차의 circular moving-block bootstrap 기간 합계.

    동일한 날짜 인덱스를 모든 정책 비교에 적용한다. 마지막 날 다음을 첫날로
    연결해 시작 위치를 균등 추출하며 원래 일수만큼 잘라 합산한다.
    """
    values = np.asarray(daily_differences, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("daily_differences must be a finite nonempty 2D array")
    n = len(values)
    if not 1 <= block_days <= n or n_resamples < 1:
        raise ValueError("invalid block length or resample count")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_resamples, int(np.ceil(n / block_days))))
    indices = ((starts[..., None] + np.arange(block_days)) % n).reshape(n_resamples, -1)[:, :n]
    return values[indices].sum(axis=1)


def policy_uncertainty(pred, block_lengths=(1, 3, 7), n_resamples=10000, seed=42):
    """고정 정책의 비용차를 일별로 쌍체 재표집한다. 양수는 후속 정책의 절감.

    모델·보정·정책 선택과 비용 가정의 불확실성은 포함하지 않는다.
    검토는 원래 날짜에서 할당 후 고정해 복제일 예산이 합쳐지는 문제를 피한다.
    """
    from .costs import realized_cost
    sel = pred[pred.half == "a_select"]
    params = DEFAULT_PARAMS.with_(review_budget=max(1, int(
        (sel.TransactionDT // 86400).value_counts().median() * .01)))
    pols = [policy.CostThreshold(reference_amount=float(sel.amount.mean())),
            policy.AmountAwareThreshold(), policy.WithReview(), policy.FullPolicy()]
    pairs = [("P3", "P4"), ("P4", "P5"), ("P5", "P6")]
    summaries, daily_tables = [], []
    for half in ["a_select", "b_confirm"]:
        d = pred[pred.half == half].sort_values(["TransactionDT", "TransactionID"])
        days = (d.TransactionDT // 86400).to_numpy()
        day_index = np.arange(days.min(), days.max() + 1)
        daily = pd.DataFrame(index=day_index)
        for name, pol in zip(["P3", "P4", "P5", "P6"], pols):
            action = pol.decide(d.p.to_numpy(), d.amount.to_numpy(), params, review_group=days)
            costs = realized_cost(action, d.is_fraud.to_numpy(), d.amount.to_numpy(), params)
            daily[name] = pd.Series(costs).groupby(days).sum().reindex(day_index, fill_value=0)
        differences = np.column_stack([daily[a] - daily[b] for a, b in pairs])
        daily_tables.append(daily.rename_axis("day").reset_index().assign(half=half))
        for length in block_lengths:
            samples = paired_block_bootstrap(differences, length, n_resamples, seed)
            for j, (a, b) in enumerate(pairs):
                lower, upper = np.quantile(samples[:, j], [.025, .975])
                summaries.append(dict(half=half, comparison=f"{a}-{b}", block_days=length,
                    n_days=len(daily), saving=differences[:, j].sum(), ci_low=lower, ci_high=upper,
                    positive_share=(samples[:, j] > 0).mean(), n_resamples=n_resamples, seed=seed))
    return pd.DataFrame(summaries), pd.concat(daily_tables, ignore_index=True)
