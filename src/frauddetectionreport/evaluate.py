"""평가지표 — 모델 지표와 정책 지표.

모델 지표는 '확률을 믿고 정책에 넣어도 되는가'를 확인하는 중간 절차이고,
정책 지표가 본론이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .costs import ACTION_NAMES, Action, CostParams, realized_cost
from .policy import Policy

# ============================================================ 모델 지표

def discrimination(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """변별력.

    PR-AUC를 주 지표로 쓴다. 사기율 3.5%의 불균형에서 ROC-AUC는
    진음성이 압도적이라 낙관적으로 나오고, 우리가 실제로 관심 있는
    '양성으로 뽑은 것 중 진짜 비율'은 PR 곡선이 직접 보여준다.
    """
    return {
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob),
    }


def precision_at_k(y: np.ndarray, prob: np.ndarray, k: int) -> dict[str, float]:
    """상위 k건의 정밀도/재현율.

    k는 검토 예산에 대응한다. 하루 500건을 볼 수 있다면 상위 500건의
    정밀도가 곧 검토팀의 실제 경험이다. 전체 AUC보다 운영에 가깝다.
    """
    y = np.asarray(y).astype(bool)
    idx = np.argsort(-np.asarray(prob))[:k]
    hits = y[idx].sum()
    return {
        "k": k,
        "precision_at_k": hits / max(k, 1),
        "recall_at_k": hits / max(y.sum(), 1),
    }


def calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 20) -> float:
    """ECE — 예측 확률과 실제 빈도의 가중 평균 절대 차이.

    확률값이 기대비용 식에 직접 들어가므로, 순위가 맞아도 값이 틀리면
    잘못된 결정을 내린다. 이 프로젝트에서 AUC만큼 중요한 지표.
    """
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    # 등간격이 아니라 분위수 구간. 대부분의 예측이 0 근처에 몰려 있어
    # 등간격 구간은 대부분 비어버린다.
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)

    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(ece)


def calibration_table(y: np.ndarray, prob: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """reliability diagram용 구간별 집계."""
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "mean_pred": prob[m].mean(),
                "observed": y[m].mean(),
            }
        )
    return pd.DataFrame(rows)


def calibration_by_amount(
    y: np.ndarray, prob: np.ndarray, amount: np.ndarray, n_groups: int = 5
) -> pd.DataFrame:
    """금액 구간별 보정 오차.

    전체 ECE가 낮아도 고액 구간에서 어긋나면 손실이 큰 쪽에서 정책이
    틀린다. 평균 하나로는 절대 보이지 않으므로 반드시 분리해서 본다.
    """
    amount = np.asarray(amount, dtype=float)
    q = np.quantile(amount, np.linspace(0, 1, n_groups + 1))
    idx = np.clip(np.digitize(amount, q[1:-1]), 0, n_groups - 1)

    rows = []
    for g in range(n_groups):
        m = idx == g
        if m.sum() < 100:
            continue
        rows.append(
            {
                "amount_group": g,
                "amount_range": f"{amount[m].min():.0f}–{amount[m].max():.0f}",
                "n": int(m.sum()),
                "fraud_rate": np.asarray(y)[m].mean(),
                "mean_pred": np.asarray(prob)[m].mean(),
                "ece": calibration_error(np.asarray(y)[m], np.asarray(prob)[m], n_bins=10),
            }
        )
    return pd.DataFrame(rows)


def model_report(
    y: np.ndarray, prob: np.ndarray, amount: np.ndarray | None = None, k: int = 500
) -> dict:
    """모델 지표 일괄."""
    out = discrimination(y, prob)
    out.update(precision_at_k(y, prob, k))
    out["ece"] = calibration_error(y, prob)
    out["brier"] = brier_score_loss(y, prob)
    return out


# ============================================================ 정책 지표

def cost_breakdown(
    action: np.ndarray,
    is_fraud: np.ndarray,
    amount: np.ndarray,
    p: CostParams,
) -> dict[str, float]:
    """총비용을 항목별로 분해한다.

    총비용만 보면 개선의 출처를 알 수 없다. P3->P4에서 비용이 줄었을 때
    사기 손실이 준 것인지 정상 차단이 준 것인지가 여기서 드러난다.

    분해는 **행동이 아니라 비용의 성격**을 기준으로 한다. 예컨대 AUTH의
    비용은 사기 쪽에서는 '인증을 통과당해 놓친 손실', 정상 쪽에서는
    '인증에 막혀 이탈한 손실'과 '마찰'로 나뉜다. 행동 단위로 뭉뚱그리면
    P6의 비용 구성이 실제와 어긋나 보인다.

    각 항의 합은 총비용과 일치한다.
    """
    action = np.asarray(action)
    is_fraud = np.asarray(is_fraud).astype(bool)
    amount = np.asarray(amount, dtype=float)

    legit = ~is_fraud
    a_appr = action == Action.APPROVE
    a_auth = action == Action.AUTH
    a_rev = action == Action.REVIEW

    # --- 사기를 막지 못해 발생한 손실 ---
    loss_unit = amount + p.chargeback_fee
    fraud_loss = (
        loss_unit[is_fraud & a_appr].sum()
        + p.auth_pass_fraud * loss_unit[is_fraud & a_auth].sum()
        + (1 - p.review_catch_rate) * loss_unit[is_fraud & a_rev].sum()
    )

    # --- 정상 거래를 막아 발생한 손실 (매출 + 고객 이탈) ---
    block_unit = p.margin * amount + p.false_block_penalty
    false_block_loss = (
        block_unit[legit & (action == Action.BLOCK)].sum()
        + (1 - p.auth_pass_legit) * block_unit[legit & a_auth].sum()
    )

    # --- 검토 인건비 (사기/정상 양쪽) + 정상 고객이 겪는 지연 ---
    review_cost = p.review_cost * a_rev.sum() + p.review_delay_cost * (legit & a_rev).sum()

    # --- 인증 마찰 (통과 여부와 무관하게 발생) ---
    auth_friction = p.auth_friction * a_auth.sum()

    return {
        "fraud_loss": float(fraud_loss),
        "false_block_loss": float(false_block_loss),
        "review_cost": float(review_cost),
        "auth_friction": float(auth_friction),
        "total": float(realized_cost(action, is_fraud, amount, p).sum()),
    }


def operational_metrics(
    action: np.ndarray, is_fraud: np.ndarray, amount: np.ndarray, p: CostParams
) -> dict[str, float]:
    """운영 지표. 총비용은 요약이라 운영자가 쓸 수 없다."""
    action = np.asarray(action)
    is_fraud = np.asarray(is_fraud).astype(bool)
    amount = np.asarray(amount, dtype=float)
    n = len(action)

    blocked = action == Action.BLOCK
    fraud_amt = amount[is_fraud].sum()
    catch_weight = np.select(
        [
            action == Action.BLOCK,
            action == Action.REVIEW,
            action == Action.AUTH,
        ],
        [1.0, p.review_catch_rate, 1.0 - p.auth_pass_fraud],
        default=0.0,
    )
    expected_caught = catch_weight[is_fraud].sum()
    expected_caught_amt = (amount[is_fraud] * catch_weight[is_fraud]).sum()

    return {
        "block_rate": blocked.mean(),
        "auth_rate": (action == Action.AUTH).mean(),
        "review_rate": (action == Action.REVIEW).mean(),
        "review_count": int((action == Action.REVIEW).sum()),
        # 고객 경험의 직접 지표: 정상 거래 중 몇 %가 막혔나
        "false_block_rate": blocked[~is_fraud].mean() if (~is_fraud).any() else np.nan,
        # 건수가 아닌 손실액 기준 검출률. 건수 재현율과 다르게 나온다.
        "fraud_recall_count": expected_caught / max(is_fraud.sum(), 1),
        "fraud_recall_amount": expected_caught_amt / max(fraud_amt, 1),
    }


def evaluate_policy(
    policy: Policy,
    prob: np.ndarray,
    amount: np.ndarray,
    is_fraud: np.ndarray,
    p: CostParams,
    baseline_total: float | None = None,
    review_group: np.ndarray | None = None,
) -> dict:
    """정책 하나를 평가한다."""
    action = policy.decide(prob, amount, p, review_group=review_group)

    row = {"policy": policy.name}
    row.update(cost_breakdown(action, is_fraud, amount, p))
    row.update(operational_metrics(action, is_fraud, amount, p))

    if baseline_total:
        row["saving_vs_p0"] = 1 - row["total"] / baseline_total

    for a in Action:
        row[f"n_{ACTION_NAMES[a]}"] = int((action == a).sum())
    return row


def compare_policies(
    policies: list[Policy],
    prob: np.ndarray,
    amount: np.ndarray,
    is_fraud: np.ndarray,
    p: CostParams,
    review_group: np.ndarray | None = None,
) -> pd.DataFrame:
    """정책 사다리 전체를 비교한다. 보고서의 핵심 결과표.

    절대 금액은 가정에 의존하므로 P0 대비 절감률로 정규화해 보고한다.
    """
    from .policy import ApproveAll

    base = evaluate_policy(ApproveAll(), prob, amount, is_fraud, p)["total"]
    rows = [
        evaluate_policy(
            pol, prob, amount, is_fraud, p,
            baseline_total=base, review_group=review_group,
        )
        for pol in policies
    ]
    return pd.DataFrame(rows).set_index("policy")


# ============================================================ 민감도

def sensitivity(
    policies: list[Policy],
    prob: np.ndarray,
    amount: np.ndarray,
    is_fraud: np.ndarray,
    base_params: CostParams,
    param: str,
    values,
    review_group: np.ndarray | None = None,
) -> pd.DataFrame:
    """파라미터 하나를 흔들며 정책 순위 변화를 본다.

    결론은 '이 정책이 최적'이 아니라 '조건 X에서 P5가 P4를 이긴다'는
    형태여야 하고, 그 역전점을 찾는 것이 이 함수의 목적이다.
    """
    from tqdm.auto import tqdm

    rows = []
    for v in tqdm(list(values), desc=param):
        p = base_params.with_(**{param: v})
        base = evaluate_policy(
            _ApproveAllRef(), prob, amount, is_fraud, p
        )["total"]
        for pol in policies:
            r = evaluate_policy(
                pol, prob, amount, is_fraud, p,
                baseline_total=base, review_group=review_group,
            )
            rows.append({param: v, "policy": r["policy"], "total": r["total"],
                         "saving_vs_p0": r.get("saving_vs_p0", np.nan)})
    return pd.DataFrame(rows)


def _ApproveAllRef():
    from .policy import ApproveAll

    return ApproveAll()


def find_crossover(sens: pd.DataFrame, param: str, a: str, b: str) -> float | None:
    """정책 a와 b의 순위가 뒤바뀌는 파라미터 값.

    'B가 얼마 이상이면 P5가 P4를 이기는가' 같은 질문에 답한다.
    보고서의 조건부 권고가 이 값들로 서술된다.
    """
    pa = sens[sens.policy == a].set_index(param)["total"]
    pb = sens[sens.policy == b].set_index(param)["total"]
    diff = (pa - pb).sort_index()
    sign = np.sign(diff.to_numpy())
    flips = np.flatnonzero(np.diff(sign) != 0)
    if len(flips) == 0:
        return None
    i = flips[0]
    return float(diff.index[i + 1])


# ============================================================ 공정성

def fairness_check(
    action: np.ndarray, is_fraud: np.ndarray, segment: np.ndarray
) -> pd.DataFrame:
    """세그먼트별 오차단율.

    총비용이 낮아도 특정 집단이 체계적으로 과차단되면 채택할 수 없다.
    """
    action = np.asarray(action)
    is_fraud = np.asarray(is_fraud).astype(bool)
    seg = pd.Series(np.asarray(segment)).astype(str)

    rows = []
    for name, m in seg.groupby(seg).groups.items():
        m = np.asarray(m)
        legit = m[~is_fraud[m]]
        if len(legit) < 100:
            continue
        rows.append(
            {
                "segment": name,
                "n": len(m),
                "fraud_rate": is_fraud[m].mean(),
                "false_block_rate": (action[legit] == Action.BLOCK).mean(),
                "auth_rate": (action[m] == Action.AUTH).mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("false_block_rate", ascending=False)
    return out
