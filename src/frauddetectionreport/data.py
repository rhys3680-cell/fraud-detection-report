"""원본 CSV 로딩, 병합, dtype 최적화, parquet 캐시.

CSV 재파싱은 매번 1분 이상 걸린다. 00 노트북에서 한 번만 수행하고
이후 모든 노트북은 parquet을 읽는다.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    ID_COL,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET,
    TEST_PARQUET,
    TRAIN_PARQUET,
)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """메모리 사용량을 줄인다.

    434개 변수 대부분이 float64로 읽히는데, 실제 값 범위는 float32로 충분하다.
    원본 대비 절반 이하로 줄어 이후 노트북의 반복 실험이 빨라진다.

    주의: 타깃과 ID는 건드리지 않는다.
    """
    out = df.copy()
    for col in out.columns:
        if col in (TARGET, ID_COL):
            continue
        kind = out[col].dtype.kind
        if kind == "f":
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif kind in "iu":
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif kind == "O":
            # 범주형 변수는 카디널리티가 낮을 때만 category로.
            # card1처럼 1만 개 이상인 것은 category가 오히려 손해다.
            n_unique = out[col].nunique(dropna=False)
            if n_unique / max(len(out), 1) < 0.5:
                out[col] = out[col].astype("category")
    return out


def load_raw(split: str) -> pd.DataFrame:
    """transaction + identity를 병합해 읽는다.

    identity는 일부 거래에만 존재하므로 left join. 조인 후 identity 계열
    변수가 통째로 결측인 행이 다수 생기는데, 이는 결손이 아니라
    '기기 정보가 수집되지 않은 거래'라는 정보 자체다.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    tx_path = RAW_DIR / f"{split}_transaction.csv"
    id_path = RAW_DIR / f"{split}_identity.csv"
    for p in (tx_path, id_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} 없음. Kaggle에서 받아 data/raw/ 에 배치하세요."
            )

    tx = pd.read_csv(tx_path)
    idf = pd.read_csv(id_path)

    # test_identity는 컬럼명이 id-01 형태(하이픈)로 다르다. train에 맞춰 통일.
    idf = idf.rename(columns=lambda c: c.replace("-", "_"))

    return tx.merge(idf, on=ID_COL, how="left")


def build_parquet(split: str, overwrite: bool = False) -> pd.DataFrame:
    """원본을 읽어 dtype 최적화 후 parquet으로 저장하고 반환한다."""
    out_path = TRAIN_PARQUET if split == "train" else TEST_PARQUET
    if out_path.exists() and not overwrite:
        return pd.read_parquet(out_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = _downcast(load_raw(split))
    df.to_parquet(out_path, index=False)
    return df


def load(split: str) -> pd.DataFrame:
    """캐시된 parquet을 읽는다. 없으면 만든다."""
    return build_parquet(split, overwrite=False)


# ---------------------------------------------------------------- 변수군 분류

def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """익명 변수군별로 컬럼을 나눈다.

    IEEE-CIS는 변수 대부분이 익명화되어 있고 접두사로만 성격을 짐작할 수 있다.
      C : 카운팅 계열 (해당 카드의 주소 개수 등)
      D : 시간차 계열 (이전 거래로부터 경과일 등)
      M : 매칭 플래그 (이름/주소 일치 여부)
      V : Vesta 내부 파생 변수 339개
      id: identity 파일의 기기/네트워크 변수
    """
    cols = list(df.columns)

    def by_prefix(p: str) -> list[str]:
        return [c for c in cols if c.startswith(p) and c[len(p):].isdigit()]

    groups = {
        "C": by_prefix("C"),
        "D": by_prefix("D"),
        "M": by_prefix("M"),
        "V": by_prefix("V"),
        "id": [c for c in cols if c.startswith("id_")],
        "card": [c for c in cols if c.startswith("card")],
        "addr": [c for c in cols if c.startswith("addr")],
        "email": [c for c in cols if c.endswith("emaildomain")],
        "device": [c for c in cols if c.startswith("Device")],
    }
    assigned = {c for g in groups.values() for c in g}
    groups["other"] = [c for c in cols if c not in assigned]
    return groups


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 결측률과 카디널리티.

    V 변수군은 결측이 블록 단위로 발생한다 (같이 비고 같이 채워짐).
    이 구조 자체가 '어느 수집 경로를 탔는가'라는 정보다.
    """
    return pd.DataFrame(
        {
            "missing_rate": df.isna().mean(),
            "n_unique": df.nunique(dropna=True),
            "dtype": df.dtypes.astype(str),
        }
    ).sort_values("missing_rate", ascending=False)
