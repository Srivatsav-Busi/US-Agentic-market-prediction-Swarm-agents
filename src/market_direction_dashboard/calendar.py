from __future__ import annotations

from datetime import date, datetime, timedelta


def is_market_closed(date_val: str | date | datetime) -> bool:
    if isinstance(date_val, str):
        dt = datetime.fromisoformat(date_val).date()
    elif isinstance(date_val, datetime):
        dt = date_val.date()
    else:
        dt = date_val

    # Weekends
    if dt.weekday() >= 5:
        return True

    # US Market Holidays (NYSE/NASDAQ)
    holidays = get_market_holidays(dt.year)
    if dt in holidays:
        return True

    return False


def get_market_holidays(year: int) -> set[date]:
    """Returns a set of dates for US market holidays in a given year."""
    holidays = set()

    # New Year's Day (Jan 1, or Friday before if Sat, Monday after if Sun)
    holidays.add(_observe_holiday(date(year, 1, 1)))

    # Martin Luther King Jr. Day (Third Monday in January)
    holidays.add(_nth_weekday(year, 1, 0, 3))

    # Washington's Birthday / Presidents Day (Third Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday (Friday before Easter Sunday)
    holidays.add(get_easter_sunday(year) - timedelta(days=2))

    # Memorial Day (Last Monday in May)
    holidays.add(_last_weekday(year, 5, 0))

    # Juneteenth (June 19, or Friday before if Sat, Monday after if Sun)
    # Note: Market closure for Juneteenth started in 2022
    if year >= 2022:
        holidays.add(_observe_holiday(date(year, 6, 19)))

    # Independence Day (July 4, or Friday before if Sat, Monday after if Sun)
    holidays.add(_observe_holiday(date(year, 7, 4)))

    # Labor Day (First Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving Day (Fourth Thursday in November)
    holidays.add(_nth_weekday(year, 11, 3, 4))

    # Christmas Day (Dec 25, or Friday before if Sat, Monday after if Sun)
    holidays.add(_observe_holiday(date(year, 12, 25)))

    return holidays


def _observe_holiday(dt: date) -> date:
    """Follows standard US market holiday observance rules."""
    if dt.weekday() == 5:  # Saturday -> Friday before
        return dt - timedelta(days=1)
    if dt.weekday() == 6:  # Sunday -> Monday after
        return dt + timedelta(days=1)
    return dt


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Finds the nth occurrence of a weekday in a month."""
    first_day = date(year, month, 1)
    # weekday is 0=Mon, 6=Sun
    offset = (weekday - first_day.weekday() + 7) % 7
    return first_day + timedelta(days=offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Finds the last occurrence of a weekday in a month."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday + 7) % 7
    return last_day - timedelta(days=offset)


def get_easter_sunday(year: int) -> date:
    """Calculates Easter Sunday using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
