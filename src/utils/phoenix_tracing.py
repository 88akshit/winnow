"""Optional Phoenix/OpenInference tracing for Winnow's LangChain workflow."""

import os
from contextlib import contextmanager
from typing import Iterator

from src.utils.logging_config import get_logger

logger = get_logger("winnow.phoenix")
_tracer_provider = None
_setup_attempted = False


def phoenix_enabled() -> bool:
    return os.getenv("WINNOW_PHOENIX_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def initialize_phoenix() -> bool:
    """Register a Phoenix exporter and LangChain instrumentor once per process."""
    global _setup_attempted, _tracer_provider
    if not phoenix_enabled():
        return False
    if _tracer_provider is not None:
        return True
    if _setup_attempted:
        return False
    _setup_attempted = True

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        _tracer_provider = register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "winnow"),
            endpoint=os.getenv(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
            ),
            auto_instrument=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
        logger.info(
            "event=phoenix_tracing_enabled project=%s",
            os.getenv("PHOENIX_PROJECT_NAME", "winnow"),
        )
        return True
    except Exception as exc:
        logger.warning("event=phoenix_tracing_unavailable error_type=%s", type(exc).__name__)
        _tracer_provider = None
        return False


@contextmanager
def phoenix_trace_context(thread_id: str, entrypoint: str) -> Iterator[None]:
    """Attach a session ID and entrypoint tags to traces in the current context."""
    if not initialize_phoenix():
        yield
        return

    try:
        from openinference.instrumentation import using_session, using_tags
    except ImportError as exc:
        logger.warning("event=phoenix_context_unavailable error_type=%s", type(exc).__name__)
        yield
        return
    with using_session(thread_id), using_tags(["winnow", entrypoint]):
        yield


def flush_phoenix() -> None:
    """Flush buffered traces during orderly process shutdown."""
    if _tracer_provider is None:
        return
    try:
        _tracer_provider.force_flush()
    except Exception as exc:
        logger.warning("event=phoenix_flush_failed error_type=%s", type(exc).__name__)
