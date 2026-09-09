# 04. 피처 설계 보고서

## 핵심 질문:

정책의 기대비용 계산에 사용할 만큼 확률이 정확한가?

### 비가중 LGBM, 가중 LGBM, LR 비교

### PR-AUC와 ROC-AUC

### Precision/Recall@500

### raw/isotonic/Platt 선택 과정

### test 보정 결과

### 금액 구간별 ECE

### UID ablation

### 피처 중요도와 SHAP

## 현재 핵심 결과

- 비가중 LGBM이 가장 우수
- calib에서는 isotonic 선택
- test에서는 raw가 PR-AUC와 정책 비용에서 더 좋음
- 고액 구간은 isotonic 보정 후 ECE 악화
- UID 효과는 작고 혼합적

## 핵심 결론

비가중 LGBM을 사용하되, 보정 성능이 기간에 따라 변하므로 보정법의 정기적인 재검증이 필요하다.

## 중요

- test 결과를 보고 raw로 교체하면 안 됨
- 대신 "calib에서 선택한 isotonic의 미래 안정성이 낮았다"를 발견으로 보고
