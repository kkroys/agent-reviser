import os
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracer(Protocol):
    def __call__(self, **kwargs):
        ...


class NoOpTracer:
    def __call__(self, **kwargs):
        def decorator(func):
            return func

        return decorator


def is_langsmith_enabled():
    return (os.getenv("LANGCHAIN_TRACING_V2") == "true" or
            os.getenv("LANGCHAIN_API_KEY") is not None)


def get_tracer():
    if is_langsmith_enabled():
        try:
            from langsmith import traceable
            logger.info("LangSmith tracing is enabled.")
            return traceable
        except ImportError:
            logger.warning("LangSmith is configured but the required packages are not installed.")
            return NoOpTracer()
    else:
        logger.info("LangSmith tracing is not enabled.")
        return NoOpTracer()


def get_openai_client(api_key: str):
    if is_langsmith_enabled():
        try:
            from langsmith.wrappers import wrap_openai
            from openai import OpenAI
            return wrap_openai(OpenAI(api_key=api_key))
        except ImportError:
            logger.warning("LangSmith is enabled but wrap_openai is not available.")
            return None
    return None


tracer = get_tracer()
