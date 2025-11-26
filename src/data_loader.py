import numpy as np
import pandas as pd

MONTH_CODE = {
    1:"F", 2:"G", 3:"H", 4:"J", 5:"K", 6:"M",
    7:"N", 8:"Q", 9:"U", 10:"V", 11:"X", 12:"Z"
}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODE.items()}

def get_futures_symbol_for_fomc(meeting_date: pd.Timestamp) -> str:
    """
    FOMC 회의 날짜로부터 해당 월의 ZQ 선물 심볼을 생성합니다.
    예: 2025-03-18 → ZQH2025
    """
    if not isinstance(meeting_date, pd.Timestamp):
        meeting_date = pd.to_datetime(meeting_date)

    year = meeting_date.year
    month = meeting_date.month
    mcode = MONTH_CODE[month]
    return f"ZQ{mcode}{year}"

def get_next_futures_symbol(meeting_date: pd.Timestamp) -> str:
    """
    FOMC 회의가 있는 달의 '다음 달' ZQ 선물 심볼을 생성합니다.
    예: 2025-03-18 → ZQJ2025 (4월물)
        2025-12-10 → ZQF2026 (다음 해 1월물)
    """
    if not isinstance(meeting_date, pd.Timestamp):
        meeting_date = pd.to_datetime(meeting_date)

    year = meeting_date.year
    month = meeting_date.month
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    mcode = MONTH_CODE[month]
    return f"ZQ{mcode}{year}"

def get_next_symbol_from_front(sym_front: str) -> str:
    """
    'ZQH2025' 같은 front 월물 심볼을 받아
    바로 다음 달 월물 심볼을 반환합니다.
    예: ZQX2025 (11월물) -> ZQZ2025 (12월물)
        ZQZ2025 (12월물) -> ZQF2026 (다음해 1월물)
    """
    # 형식: ZQ + month_code + year
    prefix = sym_front[:2]      # 'ZQ'
    mcode = sym_front[2]        # 'H', 'M', 'Z' ...
    year = int(sym_front[3:])   # '2025' -> 2025

    month = CODE_TO_MONTH[mcode]
    if month == 12:
        month_next = 1
        year_next = year + 1
    else:
        month_next = month + 1
        year_next = year

    mcode_next = MONTH_CODE[month_next]
    return f"{prefix}{mcode_next}{year_next}"

def get_front_symbol_for_meeting(meeting_date: pd.Timestamp) -> str:
    """
    단순 버전: 회의가 속한 '달'의 월물 = front 로 사용
    예: 2025-10-29 -> ZQV2025 (10월물)
    """
    if not isinstance(meeting_date, pd.Timestamp):
        meeting_date = pd.to_datetime(meeting_date)
    year = meeting_date.year
    month = meeting_date.month
    mcode = MONTH_CODE[month]
    return f"ZQ{mcode}{year}"