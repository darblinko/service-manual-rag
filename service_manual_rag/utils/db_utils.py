"""Retry helper for database clients."""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10,
    exceptions: tuple = (Exception,),
):
    """Retry the decorated function with exponential backoff on the given exceptions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as error:
                logger.warning(f"Retrying {func.__name__} after {type(error).__name__}: {error}")
                raise

        return wrapper

    return decorator
