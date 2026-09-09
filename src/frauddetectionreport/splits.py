"""시간 기반 분할.

사기 수법은 시간에 따라 변한다. 랜덤 split은 반복 개체와 여러 기간을
학습·평가 양쪽에 섞어 운영 시점보다 낙관적일 수 있다. 대회 test set도 train 이후 기간이므로
시간 순서를 지키는 것이 평가의 전제다.

    시간축 ────────────────────────────────────────────►
    [      train      ][  calib  ][   test (정책 평가)   ]

- train : 모델 학습, 하이퍼파라미터 탐색(내부 시간 CV)
- calib : 확률 보정 (학습 구간과 겹치면 안 됨)
- test  : 정책 비교. 정책 선택 과적합을 막기 위해 다시 전/후반으로 나눈다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TIME_COL

# 기본 비율. train 구간을 넉넉히 두되 보정법의 fit/선택에 충분한 calib을 남긴다.
DEFAULT_TRAIN_FRAC = 0.65
DEFAULT_CALIB_FRAC = 0.15


@dataclass(frozen=True)
class TimeSplit:
    """분할 경계와 마스크를 담는다."""

    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    cut_train: float
    cut_calib: float

    def summary(self, df: pd.DataFrame, target: str = "isFraud") -> pd.DataFrame:
        """구간별 건수·기간·사기율. 드리프트 점검용."""
        rows = []
        for name, mask in (("train", self.train), ("calib", self.calib), ("test", self.test)):
            sub = df.loc[mask]
            rows.append(
                {
                    "split": name,
                    "n": len(sub),
                    "days": (sub[TIME_COL].max() - sub[TIME_COL].min()) / 86_400,
                    "fraud_rate": sub[target].mean() if target in sub else np.nan,
                }
            )
        return pd.DataFrame(rows)


def make_time_split(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    calib_frac: float = DEFAULT_CALIB_FRAC,
) -> TimeSplit:
    """시간 분위수로 3분할한다.

    건수가 아니라 시각 기준으로 자른다. 거래량은 기간마다 다르므로
    시각 기준이라야 '이 시점 이후는 미래'라는 의미가 정확해진다.
    """
    t = df[TIME_COL].to_numpy()
    cut_train = float(np.quantile(t, train_frac))
    cut_calib = float(np.quantile(t, train_frac + calib_frac))

    return TimeSplit(
        train=t <= cut_train,
        calib=(t > cut_train) & (t <= cut_calib),
        test=t > cut_calib,
        cut_train=cut_train,
        cut_calib=cut_calib,
    )


def split_test_halves(df: pd.DataFrame, test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """test를 전반/후반으로 나눈다.

    여러 정책을 test에서 비교해 최고를 고르면 test에 과적합된다.
    앞 절반에서 정책을 선택하고 뒤 절반에서 확인한다.
    드리프트 하에서 정책이 유지되는지도 함께 볼 수 있다.
    """
    t = df[TIME_COL].to_numpy()
    mid = float(np.quantile(t[test_mask], 0.5))
    return test_mask & (t <= mid), test_mask & (t > mid)


def time_series_folds(
    df: pd.DataFrame, mask: np.ndarray, n_folds: int = 4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """train 구간 내부의 시간 기반 CV 폴드.

    확장 윈도우 방식: 각 폴드의 검증 구간은 항상 학습 구간 이후다.
    하이퍼파라미터 탐색에만 쓰고, calib/test는 건드리지 않는다.
    """
    t = df[TIME_COL].to_numpy()
    idx_time = t[mask]
    bounds = np.quantile(idx_time, np.linspace(0, 1, n_folds + 2)[1:-1])

    folds = []
    for i, b in enumerate(bounds):
        nxt = bounds[i + 1] if i + 1 < len(bounds) else idx_time.max() + 1
        folds.append((mask & (t <= b), mask & (t > b) & (t <= nxt)))
    return folds


def random_split_for_comparison(
    df: pd.DataFrame, train_frac: float = DEFAULT_TRAIN_FRAC, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """랜덤 split — **비교 목적으로만** 사용한다.

    03 노트북에서 시간 split 대비 성능이 얼마나 부풀려지는지 보이는 데 쓴다.
    실제 모델링에는 절대 쓰지 않는다.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    k = int(n * train_frac)
    tr = np.zeros(n, dtype=bool)
    tr[perm[:k]] = True
    return tr, ~tr
