import numpy as np
import pandas as pd
import time

from tvDatafeed import TvDatafeed, Interval

from config.account_info import *
from src.data_loader import *
from src.utils import *
from tqdm import tqdm

from websocket._exceptions import WebSocketTimeoutException

def safe_get_hist(tv, symbol, exchange, interval, n_bars, max_retries=5, sleep_sec=1.0):
    """
    TradingView tv.get_hist() 호출 시 websocket timeout이 자주 나는 문제 방지용.
    일정 횟수까지 재시도하고 실패하면 None 반환.
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=n_bars,
            )
            if df is not None and not df.empty:
                return df
            else:
                # 데이터 없음도 예외처럼 처리하여 재시도
                raise ValueError("No data returned")

        except (WebSocketTimeoutException, ConnectionError, TimeoutError, Exception) as e:
            print(f"[Retry {attempt}/{max_retries}] {symbol} {exchange} 오류: {type(e).__name__}: {e}")
            time.sleep(sleep_sec)

    print(f"failed: {symbol} ({exchange}), try {max_retries} times...")
    return None

def get_interest_probs_single_contract(
        futures_prices: pd.Series,
        effr_series: pd.Series,
        bp_moves: list[float] | np.ndarray,
        meeting_date: pd.Timestamp,
        expiry_date: pd.Timestamp | None = None,
        ma_window: int | None = 5,
        ma_apply_days: int | None = 20,
    ) -> pd.DataFrame:
    """
    단일 ZQ 월물 + EFFR + FOMC 회의 날짜를 이용해
    임의의 bp 시나리오들에 대한 확률 시계열을 계산합니다.

    특징
    ----
    - 월평균 금리 계산은 'meeting_date' 기준 (FedWatch 스타일)
    - 선물 가격 smoothing 은 '선물 만기(expiry_date)'까지 남은 일수 기준으로 적용

    Parameters
    ----------
    futures_prices : pd.Series
        index = DatetimeIndex, values = ZQ 선물 종가
    effr_series : pd.Series
        index = DatetimeIndex, values = EFFR (%, 예 3.83)
    bp_moves : list[float] or np.ndarray
        [-25, 0, 25] 등 FOMC에서의 변동 bp 시나리오
    meeting_date : pd.Timestamp
        해당 FOMC 회의 날짜 (월평균 가중치 계산용)
    expiry_date : pd.Timestamp or None
        이 선물 월물의 만기일.
        None 이면 futures_prices.index.max() 를 만기일로 근사.
    ma_window : int or None
        선물 가격 smoothing 에 사용할 이동평균 window (trailing).
        None 또는 1 이면 smoothing 사용 안 함.
    ma_apply_days : int or None
        만기일까지 남은 일수가 이 값 이하인 날짜들에만 smoothing 적용.
        예: 20 → 만기 20일 전부터 이동평균 사용.
        None 이면 전체 구간에 smoothing 적용.

    Returns
    -------
    pd.DataFrame
        index  = futures_prices.index
        columns:
            - 'price'      : 실제 확률 계산에 사용된 선물 가격 (raw or MA)
            - 'impl_rate'  : 100 - price
            - 'prob_[bp]'  : 각 bp 시나리오에 대한 확률 열 (예: prob_-25, prob_0, prob_25)
    """
    # 타입 정리
    if not isinstance(meeting_date, pd.Timestamp):
        meeting_date = pd.to_datetime(meeting_date)

    prices = futures_prices.sort_index()
    effr_aligned = effr_series.sort_index().reindex(prices.index, method="ffill")

    # 선물 만기일 설정 (없으면 가격 시계열 마지막 날짜로 근사)
    if expiry_date is None:
        expiry_date = prices.index.max()
    else:
        if not isinstance(expiry_date, pd.Timestamp):
            expiry_date = pd.to_datetime(expiry_date)

    # 월평균 가중치 (회의 기준)
    N, N_before, N_after = compute_month_weights(meeting_date)

    # bp 시나리오 설정
    bp_moves = np.asarray(bp_moves, dtype=float)
    delta_bp = bp_moves / 100.0   # 25bp -> 0.25

    # 선물 가격 rolling mean (전체 시리즈 기준)
    if ma_window is not None and ma_window > 1:
        prices_ma = prices.rolling(window=ma_window, min_periods=1).mean()
    else:
        prices_ma = prices

    n_dates = len(prices)
    n_scen  = len(bp_moves)

    prob_mat   = np.zeros((n_dates, n_scen), dtype=float)
    impl_rates = np.empty(n_dates, dtype=float)
    used_prices = np.empty(n_dates, dtype=float)

    index_list = prices.index.to_list()

    for i, dt in enumerate(index_list):
        r0 = effr_aligned.loc[dt]

        # 만기일까지 남은 일수 기준 smoothing 적용 여부 결정
        days_to_expiry = (expiry_date - dt).days

        if ma_window is not None and ma_window > 1:
            if (ma_apply_days is None) or (days_to_expiry <= ma_apply_days):
                px = prices_ma.loc[dt]
            else:
                px = prices.loc[dt]
        else:
            px = prices.loc[dt]

        used_prices[i] = px
        r_impl = 100.0 - px
        impl_rates[i] = r_impl

        # 각 bp 시나리오에 대한 회의 후 금리 & 월평균 금리
        post_rates = r0 + delta_bp                     # (K,)
        avg_rates  = (N_before * r0 + N_after * post_rates) / N

        # 월평균 금리 기준으로 정렬
        order = np.argsort(avg_rates)
        avg_sorted = avg_rates[order]
        probs_sorted = np.zeros_like(avg_sorted)

        # 선형 보간으로 확률 분배
        if r_impl <= avg_sorted[0]:
            probs_sorted[0] = 1.0
        elif r_impl >= avg_sorted[-1]:
            probs_sorted[-1] = 1.0
        else:
            idx = np.searchsorted(avg_sorted, r_impl) - 1
            idx = max(0, min(idx, len(avg_sorted) - 2))

            r_low  = avg_sorted[idx]
            r_high = avg_sorted[idx + 1]
            denom = r_high - r_low

            if abs(denom) < 1e-12:
                probs_sorted[idx] = 1.0
            else:
                w_high = (r_impl - r_low) / denom
                w_low  = 1.0 - w_high
                probs_sorted[idx]     = w_low
                probs_sorted[idx + 1] = w_high

        s = probs_sorted.sum()
        if s > 0:
            probs_sorted /= s

        # 정렬 이전 bp 순서로 복원
        inv_order = np.argsort(order)
        prob_mat[i, :] = probs_sorted[inv_order]

    # 결과 DataFrame 구성
    prob_cols = {
        f"prob_{int(bp)}": prob_mat[:, j]
        for j, bp in enumerate(bp_moves)
    }

    out = pd.DataFrame(
        {
            "price": used_prices,
            "impl_rate": impl_rates,
            **prob_cols,
        },
        index=prices.index,
    )

    return out

def get_interest_probs_blended_two_contracts(
        front_prices: pd.Series,  # meeting 달 ZQ
        next_prices: pd.Series,   # 다음 달 ZQ
        effr_series: pd.Series,
        bp_moves: list[float] | np.ndarray,
        meeting_date: pd.Timestamp,
        taper_start: int = 20,
        taper_end: int = 0,
        w_far: float = 1.0,
        w_near: float = 0.2,
    ) -> pd.DataFrame:
    """
    front 월물 + next 월물 두 개를 사용해
    날짜별 확률을 가중 합(blending)으로 계산합니다.

    - 먼 시점: front 비중 1.0
    - 회의에 가까워질수록 front 비중을 줄이고 next 비중을 키움
    """
    # 1) 각각 단일월물 기준 확률 계산
    df_front = get_interest_probs_single_contract(
        futures_prices=front_prices,
        effr_series=effr_series,
        bp_moves=bp_moves,
        meeting_date=meeting_date,
    )

    # next 월물이 없거나 데이터가 비어 있으면 front만 사용
    if next_prices is None or len(next_prices) == 0:
        return df_front

    df_next = get_interest_probs_single_contract(
        futures_prices=next_prices,
        effr_series=effr_series,
        bp_moves=bp_moves,
        meeting_date=meeting_date,
    )

    # 2) 공통 index 기준으로 align (outer join 후, 필요 시 ffill도 가능)
    idx = df_front.index.union(df_next.index).sort_values()
    df_f = df_front.reindex(idx)
    df_n = df_next.reindex(idx)

    # 3) 각 날짜별 front weight 계산
    weights = []
    for d in idx:
        w = front_weight_by_days_to_meeting(
            current_date=d,
            meeting_date=meeting_date,
            taper_start=taper_start,
            taper_end=taper_end,
            w_far=w_far,
            w_near=w_near,
        )
        weights.append(w)
    w_front = np.asarray(weights)
    w_next = 1.0 - w_front

    # 4) blended 확률 계산
    prob_cols = [c for c in df_f.columns if c.startswith("prob_")]

    blended_data = {}
    # price, impl_rate도 참고용으로 섞어줌 (그냥 가중평균)
    if "price" in df_f.columns and "price" in df_n.columns:
        blended_data["price"] = w_front * df_f["price"].values + w_next * df_n["price"].values
    if "impl_rate" in df_f.columns and "impl_rate" in df_n.columns:
        blended_data["impl_rate"] = w_front * df_f["impl_rate"].values + w_next * df_n["impl_rate"].values

    for col in prob_cols:
        pf = df_f[col].values
        pn = df_n[col].values
        blended_data[col] = w_front * pf + w_next * pn

    out = pd.DataFrame(blended_data, index=idx)

    return out

def build_fomc_prob_time_series(
        fomc_dates: list[pd.Timestamp] | list[str],
        effr_series: pd.Series,
        bp_moves: list[float] | np.ndarray = (-25, 0, 25),
        start_date: str | pd.Timestamp = "2000-01-01",
        end_date: str | pd.Timestamp | None = None,
        ma_window: int | None = 5,
        ma_apply_days: int | None = 20,
        tv_username=None,
        tv_password=None,
    ) -> pd.DataFrame:
    """
    FOMC dates, EFFR, ZQ 선물을 이용해 '다가오는 FOMC' 기준 인하/동결/인상 확률 시계열 생성.
    (단일 월물 + 만기일 smoothing)
    """

    # fomc_dates 정렬
    fomc_dates = [pd.to_datetime(d) for d in fomc_dates]
    fomc_dates = sorted(fomc_dates)

    if end_date is None:
        end_date = pd.Timestamp.today().normalize()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    all_days = pd.date_range(start_date, end_date, freq="D")

    # 로그인 정보 적용
    tv = TvDatafeed(tv_username, tv_password)

    probs_by_meeting: dict[pd.Timestamp, pd.DataFrame] = {}

    for md in tqdm(fomc_dates):
        # 회의가 있는 달의 ZQ 월물 심볼
        sym_front = get_futures_symbol_for_fomc(md)

        # TradingView에서 front 월물 가져오기 (재시도 포함)
        zq_front = safe_get_hist(
            tv,
            symbol=sym_front,
            exchange="CBOT",
            interval=Interval.in_daily,
            n_bars=5000,
            max_retries=5,
            sleep_sec=1.0,
        )
        if zq_front is None:
            continue

        zq_front.index = pd.to_datetime(zq_front.index.strftime("%Y-%m-%d"))

        # 월말 BMonthEnd 기준 선물 만기일
        contract_month_start = md.replace(day=1)
        expiry_date = contract_month_start + pd.offsets.BMonthEnd(0)

        # 단일 월물 기반 확률 계산
        df_probs = get_interest_probs_single_contract(
            futures_prices=zq_front["close"],
            effr_series=effr_series,
            bp_moves=bp_moves,
            meeting_date=md,
            expiry_date=expiry_date,
            ma_window=ma_window,
            ma_apply_days=ma_apply_days,
        )

        probs_by_meeting[md] = df_probs

    # 날짜별로 '다가오는 첫 FOMC' 기준 확률 선택
    records = []
    fomc_array = np.array(fomc_dates)

    for d in all_days:
        idx = np.searchsorted(fomc_array, d, side="left")
        if idx >= len(fomc_array):
            break

        md = fomc_array[idx]
        df_m = probs_by_meeting.get(md)
        if df_m is None:
            continue

        if d not in df_m.index:
            continue

        row = df_m.loc[d]

        rec = {
            "date": d,
            "meeting_date": md,
        }
        for col in row.index:
            if col.startswith("prob_"):
                rec[col] = row[col]
        records.append(rec)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("date").sort_index()
    return result