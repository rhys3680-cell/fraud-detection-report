"""피처 엔지니어링.

IEEE-CIS는 모델보다 피처가 중요한 데이터다. 효과 순서:

  1. 카드 소유자 추정  — 가장 큰 이득. 목적 진술의 '이용자 맥락'
  2. 빈도 인코딩       — 고카디널리티 범주형
  3. 금액 파생         — 소수부, 소유자 평균 대비 비율
  4. 집계 피처         — 소유자/기기/도메인별 통계

**누수 주의**: 학습 행은 각 거래 직전까지의 이력으로 집계하고,
이후 구간에는 train 구간에서 고정한 통계를 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AMOUNT_COL, TIME_COL

UID_COL = "uid"


# ------------------------------------------------- 1. 카드 소유자 추정

def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """카드 소유자를 추정해 `uid` 를 만든다.

    IEEE-CIS에는 고객 ID가 없고 D1의 정확한 기준 개체도 공개되지 않았다.
    거래일에서 D1을 뺀 기준일과 card1·addr1을 결합한 값이 동일 이용자를
    근사하는 데 유용하다는 경쟁 사례를 따라 휴리스틱 UID로 사용한다.

        account_day = floor(TransactionDT / 86400) - D1

    여기에 card1(카드 식별자)과 addr1(청구 주소)을 결합하면
    동일 소유자를 묶을 수 있다.

    한계: 추정이며 검증할 정답이 없다. 서로 다른 사람이 같은 키를
    가질 수도, 같은 사람이 카드를 바꿔 분리될 수도 있다.
    보고서에 이 한계를 명시한다.

    결측 처리: D1, card1, addr1 중 하나라도 없으면 소유자를 추정할 수 없다. 이때
    결측을 하나의 값으로 묶으면 서로 무관한 거래들이 한 소유자가 되어
    집계 피처가 오염되므로, **거래마다 고유한 키를 주어 분리**한다.
    """
    out = df.copy()
    day = (out[TIME_COL] // 86400).astype("float32")
    account_day = (day - out["D1"].astype("float32")).round()

    addr = out["addr1"] if "addr1" in out.columns else pd.Series(np.nan, index=out.index)

    # NaN은 정수 캐스팅이 불가능하므로 float 상태에서 문자열로 만든다.
    # (Int32 캐스팅은 결측이 있으면 IntCastingNaNError를 낸다)
    def _key(s: pd.Series) -> pd.Series:
        return s.astype("float64").map(lambda v: "" if pd.isna(v) else f"{v:.0f}")

    uid = (
        _key(out["card1"])
        .str.cat([_key(addr), _key(account_day)], sep="_")
    )

    # 소유자 추정에 필요한 값이 하나라도 없으면 묶지 않는다.
    unknown = out["card1"].isna() | addr.isna() | account_day.isna()
    uid = uid.mask(unknown, "na_" + out.index.astype(str))

    out[UID_COL] = uid
    out["uid_is_estimated"] = (~unknown).astype("int8")
    return out


# ------------------------------------------------------ 2/3/4. 변환기

@dataclass
class FeatureBuilder:
    """fit/transform 분리 변환기.

    fit은 **train 구간에서만** 호출한다. 그래야 미래 정보가 새지 않는다.
    """

    freq_cols: list[str] = field(default_factory=list)
    agg_keys: list[str] = field(default_factory=list)

    freq_maps_: dict[str, pd.Series] = field(default_factory=dict, init=False)
    agg_maps_: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    # ---------------------------------------------------------------- fit

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        for col in self.freq_cols:
            if col in df.columns:
                self.freq_maps_[col] = df[col].astype(str).value_counts()

        for key in self.agg_keys:
            if key not in df.columns:
                continue
            g = df.groupby(key)[AMOUNT_COL]
            self.agg_maps_[key] = pd.DataFrame(
                {
                    f"{key}_amt_mean": g.mean(),
                    f"{key}_amt_std": g.std(),
                    f"{key}_count": g.size(),
                }
            )
        self.fitted_ = True
        return self

    # ---------------------------------------------------------- transform

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("fit()을 train 구간에서 먼저 호출하세요.")

        out = df.copy()
        out = self._add_time(out)
        out = self._add_amount(out)
        out = self._add_frequency(out)
        out = self._add_aggregates(out)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ 개별 파생

    @staticmethod
    def _add_time(df: pd.DataFrame) -> pd.DataFrame:
        """시각/요일. 01에서 사기율 패턴이 확인된 축."""
        df["hour"] = (df[TIME_COL] // 3600 % 24).astype("int8")
        df["dayofweek"] = (df[TIME_COL] // 86400 % 7).astype("int8")
        df["day"] = (df[TIME_COL] // 86400).astype("int32")
        return df

    @staticmethod
    def _add_amount(df: pd.DataFrame) -> pd.DataFrame:
        """금액 파생.

        `amt_decimal`: 소수부. 정수 금액(예: 100.00)과 환율 변환을 거친
        금액(예: 117.43)은 성격이 다르다. 실제로 강한 신호로 알려져 있다.
        """
        amt = df[AMOUNT_COL].astype("float32")
        df["amt_log"] = np.log1p(amt)
        # 결측이 있으면 정수 캐스팅이 불가능하므로 float로 둔다.
        # LightGBM은 결측을 분기로 직접 처리하므로 대치할 필요가 없다.
        df["amt_decimal"] = ((amt - np.floor(amt)) * 1000).round().astype("float32")
        df["amt_is_round"] = (df["amt_decimal"] == 0).astype("float32").mask(amt.isna())
        return df

    def _add_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """빈도 인코딩.

        희귀한 카드/도메인/기기가 사기와 연관되는 신호를 잡는다.
        train에서 못 본 범주는 0 (= 매우 희귀하다는 신호).
        """
        for col, counts in self.freq_maps_.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].astype(str).map(counts).fillna(0).astype("float32")
        return df

    def _add_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """소유자/기기별 금액 통계와 그 대비 비율.

        `_amt_ratio`가 '평소보다 큰 거래'를 나타내는 행동 신호다.
        """
        for key, table in self.agg_maps_.items():
            if key not in df.columns:
                continue
            joined = df[[key]].join(table, on=key)
            for c in table.columns:
                df[c] = joined[c].astype("float32")
            mean_col = f"{key}_amt_mean"
            df[f"{key}_amt_ratio"] = (
                df[AMOUNT_COL] / df[mean_col].replace(0, np.nan)
            ).astype("float32")
        return df


# --------------------------------------------------------- 시점 기준 집계

def add_expanding_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    """소유자별 **시점 기준 누적** 피처 — 누수 없음.

    각 거래 시점까지의 정보만 사용하므로 fit/transform 분리가 불필요하다.
    실무의 스트리밍 피처("지난 N건의 평균")를 오프라인에서 근사한 것이다.

      uid_txn_seq       : 이 소유자의 몇 번째 거래인가
      uid_secs_since    : 직전 거래로부터 경과 초 (첫 거래는 NaN)
      uid_amt_cummean   : 직전까지의 평균 금액 (현재 거래 제외)
      uid_amt_vs_hist   : 현재 금액 / 과거 평균
    """
    out = df.sort_values(TIME_COL).copy()
    g = out.groupby(UID_COL, observed=True)

    out["uid_txn_seq"] = g.cumcount().astype("int32")
    out["uid_secs_since"] = (out[TIME_COL] - g[TIME_COL].shift(1)).astype("float32")

    # shift(1)로 현재 거래를 제외해야 자기 자신을 참조하는 누수가 없다
    prev_sum = g[AMOUNT_COL].cumsum() - out[AMOUNT_COL]
    out["uid_amt_cummean"] = (prev_sum / out["uid_txn_seq"].replace(0, np.nan)).astype("float32")
    out["uid_amt_vs_hist"] = (
        out[AMOUNT_COL] / out["uid_amt_cummean"].replace(0, np.nan)
    ).astype("float32")

    return out.sort_index()


# ------------------------------------------------------------------ 조립

DEFAULT_FREQ_COLS = [
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", UID_COL,
]
DEFAULT_AGG_KEYS = [UID_COL, "card1", "DeviceInfo"]


def build_features(
    df: pd.DataFrame, fit_mask: np.ndarray
) -> tuple[pd.DataFrame, FeatureBuilder]:
    """전체 파이프라인.

    `fit_mask`는 train 구간이어야 한다. 통계를 이 구간에서만 계산해
    이후 구간에 적용함으로써 누수를 막는다.
    """
    df = add_uid(df)
    df = add_expanding_uid_features(df)

    fb = FeatureBuilder(
        freq_cols=[c for c in DEFAULT_FREQ_COLS if c in df.columns],
        agg_keys=[c for c in DEFAULT_AGG_KEYS if c in df.columns],
    )
    fb.fit(df.loc[fit_mask])
    out = fb.transform(df)

    # 학습 행에는 그 시점보다 과거인 행만으로 만든 통계를 사용한다.
    # 전체 train 통계를 같은 train 행에 붙이면 타깃 누수는 아니더라도
    # 미래 거래와 자기 자신을 참조해 운영 시점과 다른 피처가 된다.
    history = _historical_training_features(df.loc[fit_mask], fb)
    history_cols = [c for c in history.columns if c not in df.columns]
    out.loc[fit_mask, history_cols] = history[history_cols]
    return out, fb


def _historical_training_features(
    df: pd.DataFrame, fb: FeatureBuilder
) -> pd.DataFrame:
    """학습 구간의 빈도·집계를 각 행 직전 시점 기준으로 계산한다."""
    out = df.sort_values(TIME_COL).copy()
    amount = out[AMOUNT_COL].astype("float64")

    for col in fb.freq_cols:
        key = out[col].astype(str)
        out[f"{col}_freq"] = key.groupby(key, sort=False).cumcount().astype("float32")

    for key_col in fb.agg_keys:
        key = out[key_col].astype(str)
        count = key.groupby(key, sort=False).cumcount().astype("float64")
        prev_sum = amount.groupby(key, sort=False).cumsum() - amount
        prev_sq_sum = (amount.pow(2).groupby(key, sort=False).cumsum() - amount.pow(2))
        mean = prev_sum / count.replace(0, np.nan)
        variance = (prev_sq_sum - prev_sum.pow(2) / count.replace(0, np.nan)) / (count - 1)
        variance = variance.clip(lower=0).where(count > 1)

        out[f"{key_col}_count"] = count.astype("float32")
        out[f"{key_col}_amt_mean"] = mean.astype("float32")
        out[f"{key_col}_amt_std"] = np.sqrt(variance).astype("float32")
        out[f"{key_col}_amt_ratio"] = (amount / mean.replace(0, np.nan)).astype("float32")

    return out.sort_index()


def model_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """LightGBM에 넣을 컬럼. 수치형과 category만."""
    from .config import ID_COL, TARGET

    drop = {TARGET, ID_COL, UID_COL, *exclude}
    return [
        c for c in df.columns
        if c not in drop and (df[c].dtype.kind in "fiub" or df[c].dtype.name == "category")
    ]
