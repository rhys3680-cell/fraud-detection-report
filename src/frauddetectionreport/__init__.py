"""사기 탐지 정책 설계 — IEEE-CIS Fraud Detection.

사기 탐지를 분류가 아니라 **결정 문제**로 다룬다.
목표는 AUC 최대화가 아니라 총비용 최소화이며,
거래별로 승인·추가 인증·보류·차단 중 하나를 선택한다.

모듈 구성:
    config    경로와 상수
    data      로딩, 병합, parquet 캐시
    splits    시간 기반 분할
    features  피처 엔지니어링 (fit/transform 분리)
    costs     비용 구조와 기대비용  ← 목적 함수
    policy    정책 사다리 P0~P6
    evaluate  모델·정책 지표, 민감도 분석
"""

from . import config, costs, data, evaluate, features, policy, splits

__all__ = ["config", "costs", "data", "evaluate", "features", "policy", "splits"]
