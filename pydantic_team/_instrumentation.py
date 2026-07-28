"""OpenTelemetry instrumentation for team orchestration."""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import cast

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

_instrumentation_enabled: bool = False
_TRACER_NAME = 'pydantic_team'
_DEFAULT_SNIPPET_LIMIT = 120


def instrument_pydantic_team(*, enabled: bool = True) -> None:
    """Enable or disable OpenTelemetry spans for team orchestration (idempotent)."""
    global _instrumentation_enabled
    _instrumentation_enabled = enabled


def is_instrumented() -> bool:
    """Return whether team orchestration spans are currently enabled."""
    return _instrumentation_enabled


def team_span(name: str, **attrs: object) -> AbstractContextManager[None]:
    """Context manager for a team orchestration span.

    When instrumentation is disabled, yields a no-op context. When enabled, opens
    an OpenTelemetry span named ``name`` with the given attributes.
    """
    if not _instrumentation_enabled:
        return nullcontext()
    return _otel_team_span(name, attrs)


@contextmanager
def _otel_team_span(name: str, attrs: Mapping[str, object]) -> Generator[None]:
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as span:
        for key, value in attrs.items():
            if value is None:
                continue
            span.set_attribute(key, _otel_attribute_value(value))
        yield


def _otel_attribute_value(value: object) -> AttributeValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence = cast(Sequence[object], value)
        return [_sequence_item_as_str(item) for item in sequence]
    return str(value)


def _sequence_item_as_str(item: object) -> str:
    return str(item)


def snippet(text: str, *, limit: int = _DEFAULT_SNIPPET_LIMIT) -> str:
    """Collapse whitespace and truncate for span attribute values."""
    collapsed = ' '.join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f'{collapsed[: limit - 3]}...'
