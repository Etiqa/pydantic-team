from __future__ import annotations

from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_team import (
    CollaborativeTeam,
    HierarchicalTeam,
    instrument_pydantic_team,
    is_instrumented,
)
from pydantic_team._instrumentation import snippet, team_span

_SPAN_EXPORTER = InMemorySpanExporter()
_tracer_provider_configured = False


def _disable_instrumentation() -> None:
    instrument_pydantic_team(enabled=False)


def _span_exporter() -> InMemorySpanExporter:
    """Return a cleared in-memory exporter bound to the process TracerProvider.

    OpenTelemetry allows setting the global TracerProvider only once per process.
    """
    global _tracer_provider_configured
    if not _tracer_provider_configured:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))
        trace.set_tracer_provider(provider)
        _tracer_provider_configured = True
    _SPAN_EXPORTER.clear()
    return _SPAN_EXPORTER


def test_instrument_pydantic_team_is_idempotent() -> None:
    _disable_instrumentation()
    assert not is_instrumented()
    instrument_pydantic_team()
    assert is_instrumented()
    instrument_pydantic_team()
    assert is_instrumented()


def test_instrument_pydantic_team_can_disable() -> None:
    _disable_instrumentation()
    instrument_pydantic_team()
    assert is_instrumented()
    instrument_pydantic_team(enabled=False)
    assert not is_instrumented()


def test_team_span_noop_when_not_instrumented() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    with team_span('hierarchical.run', prompt='hello'):
        pass
    assert exporter.get_finished_spans() == ()


def test_team_span_emits_when_instrumented() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    with team_span(
        'hierarchical.run',
        prompt='hello',
        count=1,
        ratio=0.5,
        ok=True,
        tags=('a', 'b'),
        labels=['x', 'y'],
        skip=None,
    ):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == 'hierarchical.run'
    assert spans[0].attributes is not None
    assert spans[0].attributes['prompt'] == 'hello'
    assert spans[0].attributes['count'] == 1
    assert spans[0].attributes['ratio'] == 0.5
    assert spans[0].attributes['ok'] is True
    tags = spans[0].attributes['tags']
    assert isinstance(tags, Sequence) and not isinstance(tags, (str, bytes))
    assert list(tags) == ['a', 'b']
    labels = spans[0].attributes['labels']
    assert isinstance(labels, Sequence) and not isinstance(labels, (str, bytes))
    assert list(labels) == ['x', 'y']
    assert 'skip' not in spans[0].attributes


def test_team_span_stringifies_other_attribute_types() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    with team_span('collaborative.run', payload={'k': 1}):
        pass
    spans = exporter.get_finished_spans()
    assert spans[0].attributes is not None
    assert spans[0].attributes['payload'] == "{'k': 1}"


def test_snippet_passthrough_short() -> None:
    assert snippet('hello world') == 'hello world'


def test_snippet_collapses_whitespace() -> None:
    assert snippet('a\n\tb   c') == 'a b c'


def test_snippet_truncates_long() -> None:
    long = 'x' * 200
    result = snippet(long, limit=20)
    assert result.endswith('...')
    assert len(result) == 20


async def test_hierarchical_emits_run_and_delegate_spans() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    member = Agent(TestModel(custom_output_text='research-notes'), name='researcher')
    team = HierarchicalTeam(leader_agent=leader, members=[member])

    with leader.override(model=TestModel(custom_output_text='final-answer')):
        with member.override(model=TestModel(custom_output_text='research-notes')):
            result = await team.run('Summarize AI news')

    assert result.data == 'final-answer'
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'hierarchical.run' in names
    assert 'hierarchical.delegate' in names


async def test_hierarchical_no_spans_without_instrument() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    leader = Agent(TestModel(), name='leader')
    member = Agent(TestModel(custom_output_text='notes'), name='researcher')
    team = HierarchicalTeam(leader_agent=leader, members=[member])

    with leader.override(model=TestModel(custom_output_text='final')):
        with member.override(model=TestModel(custom_output_text='notes')):
            await team.run('hello')

    assert all(not span.name.startswith('hierarchical.') for span in exporter.get_finished_spans())


async def test_collaborative_emits_orchestration_spans() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=TestModel(custom_output_text='final-summary')):
        with worker.override(model=TestModel(custom_output_text='worker-done')):
            result = await team.run('Ship the feature')

    assert result.data == 'final-summary'
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'collaborative.run' in names
    assert 'collaborative.seed' in names
    assert 'collaborative.round' in names
    assert 'collaborative.synthesize' in names


async def test_collaborative_seed_only_when_board_complete() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=2)

    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo-lead')):
        with worker.override(model=TestModel(custom_output_text='should-not-matter')):
            result = await team.run('Nothing to split')

    assert result.data == 'solo-lead'
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'collaborative.run' in names
    assert 'collaborative.seed' in names
    assert 'collaborative.round' not in names
    assert 'collaborative.synthesize' not in names


async def test_collaborative_no_spans_without_instrument() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=TestModel(call_tools=[], custom_output_text='ok')):
        with worker.override(model=TestModel(custom_output_text='ok')):
            await team.run('x')

    assert all(not span.name.startswith('collaborative.') for span in exporter.get_finished_spans())


async def test_collaborative_max_replans_zero_skips_replan_when_incomplete() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1, max_replans=0)

    with leader.override(model=TestModel(custom_output_text='final')):
        with worker.override(model=TestModel(call_tools=[], custom_output_text='noop')):
            result = await team.run('Leave work unfinished')

    assert result.data == 'final'
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'collaborative.round' in names
    assert 'collaborative.synthesize' in names
    assert 'collaborative.replan' not in names


async def test_collaborative_streaming_dispatch_spans() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=1,
        dispatch_mode='streaming',
    )

    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo')):
        with worker.override(model=TestModel(custom_output_text='unused')):
            result = await team.run('Nothing to split')

    assert result.data == 'solo'
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'collaborative.run' in names
    assert 'collaborative.dispatch' in names
    assert 'collaborative.seed' in names
    assert 'collaborative.round' not in names


async def test_collaborative_replan_runs_when_board_incomplete() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1, max_replans=1)

    with leader.override(model=TestModel(custom_output_text='final')):
        with worker.override(model=TestModel(call_tools=[], custom_output_text='noop')):
            result = await team.run('Need more planning')

    assert result.data == 'final'
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'collaborative.replan' in names
    assert names.count('collaborative.round') >= 2
    assert 'collaborative.synthesize' in names


async def test_collaborative_replan_budget_exhausted_still_synthesizes() -> None:
    _disable_instrumentation()
    exporter = _span_exporter()
    instrument_pydantic_team()
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1, max_replans=2)

    with leader.override(model=TestModel(custom_output_text='final-after-budget')):
        with worker.override(model=TestModel(call_tools=[], custom_output_text='noop')):
            result = await team.run('Never finishes')

    assert result.data == 'final-after-budget'
    names = [span.name for span in exporter.get_finished_spans()]
    assert names.count('collaborative.replan') == 2
    assert 'collaborative.synthesize' in names
