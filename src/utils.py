import pandas as pd

def compute_month_weights(meeting_date: pd.Timestamp) -> tuple[int, int, int]:
    """
    FOMC 회의가 속한 달의
    - 전체 일수 N
    - 회의 전 일수 N_before
    - 회의 후 일수 N_after
    를 calendar day 기준으로 계산합니다.
    """
    if not isinstance(meeting_date, pd.Timestamp):
        meeting_date = pd.to_datetime(meeting_date)

    month_start = meeting_date.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)

    N = (month_end - month_start).days + 1
    N_before = (meeting_date - month_start).days
    N_after = N - N_before

    return N, N_before, N_after

def front_weight_by_days_to_meeting(
    current_date: pd.Timestamp,
    meeting_date: pd.Timestamp,
    taper_start: int = 20,
    taper_end: int = 0,
    w_far: float = 1.0,
    w_near: float = 0.2,
) -> float:
    """
    회의일까지 남은 일수에 따라 front 월물에 줄 가중치 w를 계산합니다.

    - days_to_meeting >= taper_start → w = w_far (기본 1.0)
    - days_to_meeting <= taper_end   → w = w_near (기본 0.2)
    - 그 사이는 선형 보간
    """
    if not isinstance(current_date, pd.Timestamp):
        current_date = pd.to_datetime(current_date)
    if not isinstance(meeting_date, pd.Timestamp):
        meeting_date = pd.to_datetime(meeting_date)

    days_to_meeting = (meeting_date - current_date).days

    if days_to_meeting >= taper_start:
        return w_far
    elif days_to_meeting <= taper_end:
        return w_near
    else:
        # 선형 보간
        ratio = (days_to_meeting - taper_end) / (taper_start - taper_end)
        # days_to_meeting가 taper_start일 때 1, taper_end일 때 0이 되도록
        return w_near + (w_far - w_near) * ratio