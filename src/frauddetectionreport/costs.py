"""비용 구조 — 이 프로젝트의 목적 함수.

거래 i에 대해 사기 확률 p, 금액 a 일 때 네 행동의 비용:

    행동      | 사기일 때                    | 정상일 때
    ---------|-----------------------------|--------------------------------
    승인      | a + c_cb                    | 0
    추가인증   | s_f * (a + c_cb)            | (1 - s_l) * m * a + c_auth
    보류      | ~0                          | c_rev + delta
    차단      | 0                           | m * a + L

최적 행동은 기대비용을 최소화하는 것:

    argmin_k  p * C_k^fraud + (1-p) * C_k^legit

주의 — 여기 파라미터는 **데이터에 없다**. 가정이다.
따라서 절대 금액이 아니라 상대비로 두고(차단 관련 비용을 기준으로 정규화),
결론은 민감도 분석을 통한 조건부 권고 형태로 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    """거래별 대응. 정수라 numpy 배열로 다루기 편하다."""

    APPROVE = 0
    AUTH = 1  # 추가 인증
    REVIEW = 2  # 보류 (사람 검토)
    BLOCK = 3


ACTION_NAMES = {a: a.name.lower() for a in Action}


@dataclass(frozen=True)
class CostParams:
    """비용 파라미터.

    TransactionAmt는 USD 결제금액이지만 나머지 비용 파라미터가 관측되지 않는다.
    따라서 기본값은 비교 실험을 위한 가정이며 민감도 분석을 반드시 병행한다.

    기본값은 결제업계에서 통상 인용되는 관계를 반영한 출발점이며,
    확정된 값이 아니다. 민감도 분석에서 넓게 흔들어본다.
    """

    # --- 사기 손실 ---
    chargeback_fee: float = 15.0
    """사기를 놓쳤을 때 거래 금액에 더해 발생하는 고정 처리비.
    분쟁 대응·수수료·행정 비용. 금액과 무관하게 발생한다."""

    # --- 정상 거래를 막았을 때 ---
    margin: float = 0.02
    """거래 마진율. 정상 거래를 막으면 이 만큼의 이익을 잃는다.
    금액에 비례하는 항."""

    false_block_penalty: float = 25.0
    """오차단 1건당 고객 생애가치 손상.

    이 프로젝트에서 가장 불확실하고 가장 중요한 파라미터다.
    카드가 막힌 고객은 그 카드를 덜 쓰게 되는데, 그 손실은 데이터에
    전혀 기록되지 않는다. 민감도 분석의 주 대상."""

    # --- 사람 검토 ---
    review_cost: float = 3.0
    """건당 검토 인건비."""

    review_delay_cost: float = 1.0
    """검토로 인한 지연이 정상 고객에게 주는 마찰."""

    review_catch_rate: float = 0.95
    """검토자가 사기를 실제로 잡아내는 비율.
    1.0이 아닌 이유: 사람도 오판한다. 데이터에 검토 로그가 없어 가정값."""

    # --- 추가 인증 ---
    auth_pass_fraud: float = 0.15
    """사기범이 추가 인증을 통과할 확률 (s_f).
    낮을수록 인증이 강한 필터."""

    auth_pass_legit: float = 0.90
    """정상 고객이 추가 인증을 통과할 확률 (s_l).
    통과하지 못한 고객은 이탈로 간주한다."""

    auth_friction: float = 0.5
    """인증 요구 자체의 마찰 비용 (통과 여부와 무관)."""

    # --- 운영 제약 ---
    review_budget: int | None = None
    """운영 기간별 검토 가능 건수. None이면 무제한.
    평가 시 날짜를 ``review_group``으로 넘기면 일일 예산이 된다."""

    def with_(self, **kwargs) -> "CostParams":
        """일부 값만 바꾼 사본. 민감도 분석 루프에서 쓴다."""
        return replace(self, **kwargs)

    @property
    def auth_gap(self) -> float:
        """s_l - s_f. 추가 인증의 변별력.

        0이면 인증은 아무 정보도 주지 못하고 고객만 귀찮게 하는
        순수 비용이 된다. P6의 정당성이 이 값에 달려 있다."""
        return self.auth_pass_legit - self.auth_pass_fraud


DEFAULT_PARAMS = CostParams()


# ------------------------------------------------------------- 행동별 비용

def cost_if_fraud(action: Action | np.ndarray, amount: np.ndarray, p: CostParams) -> np.ndarray:
    """거래가 실제 사기였을 때 각 행동의 비용."""
    amount = np.asarray(amount, dtype=float)
    a = np.asarray(action)

    approve = amount + p.chargeback_fee
    auth = p.auth_pass_fraud * (amount + p.chargeback_fee) + p.auth_friction
    # 검토가 완벽하지 않으므로 놓치는 비율만큼 손실이 남는다.
    review = (1 - p.review_catch_rate) * (amount + p.chargeback_fee) + p.review_cost
    block = np.zeros_like(amount)

    return np.select(
        [a == Action.APPROVE, a == Action.AUTH, a == Action.REVIEW, a == Action.BLOCK],
        [approve, auth, review, block],
        default=np.nan,
    )


def cost_if_legit(action: Action | np.ndarray, amount: np.ndarray, p: CostParams) -> np.ndarray:
    """거래가 실제 정상이었을 때 각 행동의 비용."""
    amount = np.asarray(amount, dtype=float)
    a = np.asarray(action)

    approve = np.zeros_like(amount)
    # 인증에 실패한 고객은 이탈 -> 매출 손실. 통과해도 마찰은 남는다.
    auth = (1 - p.auth_pass_legit) * (p.margin * amount + p.false_block_penalty) + p.auth_friction
    review = np.full_like(amount, p.review_cost + p.review_delay_cost)
    block = p.margin * amount + p.false_block_penalty

    return np.select(
        [a == Action.APPROVE, a == Action.AUTH, a == Action.REVIEW, a == Action.BLOCK],
        [approve, auth, review, block],
        default=np.nan,
    )


def expected_cost(
    action: Action | np.ndarray, prob: np.ndarray, amount: np.ndarray, p: CostParams
) -> np.ndarray:
    """기대비용. 결정 규칙이 최소화하는 대상."""
    prob = np.asarray(prob, dtype=float)
    return prob * cost_if_fraud(action, amount, p) + (1 - prob) * cost_if_legit(action, amount, p)


def realized_cost(
    action: np.ndarray, is_fraud: np.ndarray, amount: np.ndarray, p: CostParams
) -> np.ndarray:
    """실제 라벨로 계산한 사후 비용. 정책 평가의 기준."""
    is_fraud = np.asarray(is_fraud).astype(bool)
    return np.where(
        is_fraud,
        cost_if_fraud(action, amount, p),
        cost_if_legit(action, amount, p),
    )


# --------------------------------------------------- 최적 행동과 임계값 유도

def optimal_action(
    prob: np.ndarray,
    amount: np.ndarray,
    p: CostParams,
    allowed: tuple[Action, ...] = tuple(Action),
) -> np.ndarray:
    """기대비용을 최소화하는 행동.

    `allowed`로 행동 집합을 제한할 수 있다. 정책 사다리에서
    P3은 (APPROVE, BLOCK), P5는 REVIEW 추가, P6은 전부 사용한다.
    """
    costs = np.stack(
        [expected_cost(act, prob, amount, p) for act in allowed]
    )  # (n_actions, n)
    best = np.argmin(costs, axis=0)
    return np.asarray(allowed, dtype=int)[best]


def block_threshold(amount: np.ndarray, p: CostParams) -> np.ndarray:
    """승인 vs 차단만 있을 때 차단이 유리해지는 확률 경계.

    p*(a + c_cb) = (1-p)*(m*a + L)  을 p에 대해 풀면

        p* = (m*a + L) / (a + c_cb + m*a + L)

    금액 a가 커질수록 분자·분모 모두 커지나 분모의 a 계수(1 + m)가
    분자의 계수(m)보다 크므로 **임계값은 금액에 대해 감소**한다.
    즉 고액 거래일수록 더 낮은 확률에서도 차단이 정당화된다.
    P4(금액 의존 임계값)의 근거가 이것이다.
    """
    amount = np.asarray(amount, dtype=float)
    block_side = p.margin * amount + p.false_block_penalty
    return block_side / (amount + p.chargeback_fee + block_side)


def review_saving(prob: np.ndarray, amount: np.ndarray, p: CostParams) -> np.ndarray:
    """검토했을 때의 절감액 = (검토 안 했을 때 최선) - (검토했을 때).

    검토 예산이 유한하므로 이 값이 큰 순으로 B건만 검토한다.
    단순히 확률이 높은 순이 아니라는 점이 중요하다 — 확률이 매우 높으면
    검토 없이 차단하는 편이 낫고, 매우 낮으면 그냥 승인하는 편이 낫다.
    검토의 가치는 **결정이 애매한 구간**에서 가장 크다.
    """
    without = np.minimum(
        expected_cost(Action.APPROVE, prob, amount, p),
        expected_cost(Action.BLOCK, prob, amount, p),
    )
    with_review = expected_cost(Action.REVIEW, prob, amount, p)
    return without - with_review
