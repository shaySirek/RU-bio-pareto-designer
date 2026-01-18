from time import perf_counter
from typing import Callable, Any


def run_with_timing(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    start = perf_counter()
    result = func(*args, **kwargs)
    duration = perf_counter() - start
    return result, round(duration, 6)


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
