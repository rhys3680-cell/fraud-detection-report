"""정책 사다리 P0 ~ P6.

한 칸에 하나씩 요소를 더해, 인접 행의 차이가 곧 그 요소의 기여도가 되도록
배열했다. 개선이 멈추는 지점도 그 자체로 발견이다.

    P0  전건 승인              무개입 기준선
    P1  규칙 기반              사람의 직관
    P2  단일 임계값 차단        모델 확률의 순수 기여
    P3  비용 기반 임계값        비용에서 임계값을 유도
    P4  + 금액 의존 임계값      고액/소액 차등
    P5  + 보류 (예산 제약)      사람 검토의 한계 가치
    P6  + 추가 인증            4번째 행동

모든 정책이 같은 확률 p와 같은 비용 함수를 공유하고 **행동 규칙만** 다르다.
그래서 모델은 한 번만 학습하고 06 노트북에서 정책만 갈아끼운다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .costs import (
    Action,
    CostParams,
    block_threshold,
    expected_cost,
    optimal_action,
    review_saving,
)


class Policy(Protocol):
    """모든 정책의 공통 인터페이스."""

    name: str

    def decide(
        self,
        prob: np.ndarray,
        amount: np.ndarray,
        p: CostParams,
        review_group: np.ndarray | None = None,
    ) -> np.ndarray:
        """거래별 행동을 반환한다 (Action 정수 배열)."""
        ...


# --------------------------------------------------------------------- P0

class ApproveAll:
    """P0 — 전건 승인. 아무 대응도 하지 않았을 때의 무개입 기준선.

    모든 정책의 절감률은 이 기준 대비로 보고한다."""

    name = "P0_approve_all"

    def decide(self, prob, amount, p, review_group=None):
        return np.full(len(prob), Action.APPROVE, dtype=int)


# --------------------------------------------------------------------- P1

class RuleBased:
    """P1 — 규칙 기반. 모델 없이 사람의 직관만 사용.

    ML이 규칙보다 나은가를 확인하는 대조군이다. 실무의 규칙 엔진은
    수백 개 규칙을 운영하지만, 여기서는 비교 기준으로서 소수의
    직관적 규칙만 둔다.

    규칙은 확률을 쓰지 않고 원시 피처로만 판단한다 — 그래야
    'ML 없이 얻을 수 있는 것'의 측정이 된다.
    """

    name = "P1_rules"

    def __init__(self, amount_cut: float = 300.0, flags: np.ndarray | None = None):
        self.amount_cut = amount_cut
        self.flags = flags  # 예: 신규 기기 여부 등 04에서 만든 이진 플래그

    def decide(self, prob, amount, p, review_group=None):
        amount = np.asarray(amount, dtype=float)
        suspicious = amount >= self.amount_cut
        if self.flags is not None:
            suspicious = suspicious & np.asarray(self.flags, dtype=bool)
        return np.where(suspicious, Action.BLOCK, Action.APPROVE).astype(int)


# --------------------------------------------------------------------- P2

class SingleThreshold:
    """P2 — 단일 임계값 차단. 모델 확률을 쓰되 비용은 고려하지 않는다.

    임계값은 비용에서 유도한 것이 아니라 관행적으로 정한 값이다
    (예: 0.5, 또는 검증 구간에서 F1을 최대화하는 값).
    P3과의 차이가 '비용 구조를 반영한 것'의 기여도가 된다.
    """

    name = "P2_single_threshold"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = f"P2_threshold_{threshold:g}"

    def decide(self, prob, amount, p, review_group=None):
        return np.where(np.asarray(prob) >= self.threshold, Action.BLOCK, Action.APPROVE).astype(int)

    @classmethod
    def fit_f1(cls, prob: np.ndarray, y: np.ndarray, n_grid: int = 200) -> "SingleThreshold":
        """F1을 최대화하는 임계값. 비용을 모르는 상태의 통상적 선택을 흉내낸다."""
        from sklearn.metrics import f1_score

        grid = np.quantile(prob, np.linspace(0.5, 0.9999, n_grid))
        scores = [f1_score(y, prob >= t, zero_division=0) for t in grid]
        return cls(float(grid[int(np.argmax(scores))]))


# --------------------------------------------------------------------- P3

class CostThreshold:
    """P3 — 비용 기반 임계값 (승인/차단).

    임계값을 튜닝하지 않는다. 기대비용이 같아지는 지점을 비용 구조에서
    **유도**한다. 단, 금액을 무시하고 전체 평균 금액 기준의 단일 값을 쓴다.
    P4와의 차이가 '금액 정보'의 기여도가 된다.
    """

    name = "P3_cost_threshold"

    def __init__(self, reference_amount: float | None = None):
        self.reference_amount = reference_amount

    def decide(self, prob, amount, p, review_group=None):
        amount = np.asarray(amount, dtype=float)
        # 선택 구간에서 한 번 정한 기준 금액을 이후 확인 구간에도 고정한다.
        if self.reference_amount is None:
            self.reference_amount = float(amount.mean())
        thr = float(block_threshold(np.array([self.reference_amount]), p)[0])
        self.last_threshold = thr
        return np.where(np.asarray(prob) >= thr, Action.BLOCK, Action.APPROVE).astype(int)


# --------------------------------------------------------------------- P4

class AmountAwareThreshold:
    """P4 — 금액 의존 임계값.

    거래마다 다른 임계값을 쓴다. block_threshold()가 보이듯 임계값은
    금액에 대해 단조 감소하므로, 고액 거래는 더 낮은 확률에서도 차단된다.

    손실이 소수 고액 거래에 집중되어 있다면 P3 대비 개선이 크고,
    금액이 고르게 퍼져 있다면 차이가 작을 것이다 — 02 노트북의
    손실 집중도 분석이 이 결과를 미리 예측한다.
    """

    name = "P4_amount_aware"

    def decide(self, prob, amount, p, review_group=None):
        return optimal_action(prob, amount, p, allowed=(Action.APPROVE, Action.BLOCK))


# --------------------------------------------------------------------- P5

class WithReview:
    """P5 — 보류 추가. 검토 예산 B 제약 하의 할당 문제.

    핵심: 검토 대상을 '확률 높은 순'으로 고르지 않는다.
    확률이 매우 높으면 검토 없이 차단하는 편이 낫고, 매우 낮으면
    그냥 승인하는 편이 낫다. 검토의 가치는 **결정이 애매한 구간**에서
    가장 크므로, review_saving()이 큰 순으로 B건을 고른다.

    B가 무제한이면 순수한 기대비용 최소화와 같아진다.
    """

    name = "P5_review"

    def decide(self, prob, amount, p, review_group=None):
        prob = np.asarray(prob, dtype=float)
        base = optimal_action(prob, amount, p, allowed=(Action.APPROVE, Action.BLOCK))

        saving = review_saving(prob, amount, p)
        worth = saving > 0
        if not worth.any():
            return base

        selected = _select_reviews(saving, worth, p.review_budget, review_group)

        out = base.copy()
        out[selected] = Action.REVIEW
        return out


# --------------------------------------------------------------------- P6

class FullPolicy:
    """P6 — 네 행동 전부. 추가 인증 포함.

    추가 인증은 사실상 '거의 공짜인 두 번째 분류기'다. 모델이 p를 주고
    인증이 한 번 더 거른다. 그래서 차단하기엔 아깝고 승인하기엔 불안한
    **중간 구간**에서 최적이 된다.

    다만 s_f, s_l 은 인증을 실제로 걸어봐야 관측되는 값이고 IEEE-CIS에는
    인증 로그가 없다. 이 정책의 결과는 **데이터가 아니라 가정 위에서**
    도출된 것이며, s_l과 s_f를 함께 변화시키는 민감도 분석이 필수다.
    """

    name = "P6_full"

    def decide(self, prob, amount, p, review_group=None):
        prob = np.asarray(prob, dtype=float)
        # 보류를 뺀 세 행동으로 먼저 최적화
        base = optimal_action(
            prob, amount, p, allowed=(Action.APPROVE, Action.AUTH, Action.BLOCK)
        )

        # REVIEW를 현재 최선 행동과 비교한다. AUTH를 제외한 승인/차단과만
        # 비교하면 인증이 더 싼 거래를 검토가 덮어쓰는 오류가 생긴다.
        base_cost = expected_cost(base, prob, amount, p)
        saving = base_cost - expected_cost(Action.REVIEW, prob, amount, p)
        worth = saving > 0
        if not worth.any():
            return base

        selected = _select_reviews(saving, worth, p.review_budget, review_group)

        out = base.copy()
        out[selected] = Action.REVIEW
        return out


# --------------------------------------------------------------------- 조립

def build_ladder(
    prob_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    rule_flags: np.ndarray | None = None,
    reference_amount: float | None = None,
) -> list[Policy]:
    """P0~P6 전체를 순서대로 만든다.

    P2의 임계값은 검증 구간에서 정해야 하므로 prob_valid/y_valid를 받는다.
    없으면 0.5를 쓴다.
    """
    p2 = (
        SingleThreshold.fit_f1(prob_valid, y_valid)
        if prob_valid is not None and y_valid is not None
        else SingleThreshold(0.5)
    )
    return [
        ApproveAll(),
        RuleBased(flags=rule_flags),
        p2,
        CostThreshold(reference_amount=reference_amount),
        AmountAwareThreshold(),
        WithReview(),
        FullPolicy(),
    ]


def _select_reviews(
    saving: np.ndarray,
    worth: np.ndarray,
    budget: int | None,
    review_group: np.ndarray | None,
) -> np.ndarray:
    """기간별 절감액 상위 거래를 검토 대상으로 고른다.

    ``review_group``이 날짜라면 ``budget``은 일일 한도다. 그룹이 없으면
    입력 전체를 하나의 운영 배치로 취급한다.
    """
    if budget is None:
        return worth.copy()

    selected = np.zeros(len(saving), dtype=bool)
    if budget <= 0:
        return selected

    groups = np.zeros(len(saving), dtype=np.int8) if review_group is None else np.asarray(review_group)
    if len(groups) != len(saving):
        raise ValueError("review_group 길이는 거래 수와 같아야 합니다.")

    for group in np.unique(groups):
        cand = np.flatnonzero(worth & (groups == group))
        order = cand[np.argsort(-saving[cand], kind="stable")][:budget]
        selected[order] = True
    return selected
