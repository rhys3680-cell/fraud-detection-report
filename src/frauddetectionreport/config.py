"""프로젝트 전역 경로와 상수.

노트북 어디서 실행하든 같은 경로를 보도록 프로젝트 루트를 기준으로 잡는다.
"""

from pathlib import Path

# src/frauddetectionreport/config.py -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 노트북 간 전달되는 중간 산출물
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PARQUET = PROCESSED_DIR / "predictions.parquet"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"

# TransactionDT는 기준 시점으로부터의 초. 대회 데이터의 관례적 기준일.
# 절대 날짜 자체는 분석에 쓰지 않고, 요일/시간대 파생에만 사용한다.
DT_ORIGIN = "2017-12-01"

RANDOM_SEED = 42


def ensure_dirs() -> None:
    """산출물 디렉터리를 만든다 (이미 있으면 무시)."""
    for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
