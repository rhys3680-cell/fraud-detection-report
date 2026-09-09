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
