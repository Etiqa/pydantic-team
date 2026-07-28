from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_team._utils import (
    member_tool_description,
    member_tool_name,
    run_member,
    stringify_output,
)
from pydantic_team.base import BaseTeam, TeamResult


class _SampleModel(BaseModel):
    value: str


def test_member_tool_name_uses_agent_name() -> None:
    agent = Agent(TestModel(), name='Research Agent')
    assert member_tool_name(agent, 0) == 'Research_Agent'


def test_member_tool_name_falls_back_to_index() -> None:
    agent = Agent(TestModel())
    assert member_tool_name(agent, 2) == 'member_2'


def test_member_tool_description_uses_agent_description() -> None:
    agent = Agent(TestModel(), name='news', description='Fetch news headlines')
    assert member_tool_description(agent, 'news') == 'Fetch news headlines'


def test_member_tool_description_default() -> None:
    agent = Agent(TestModel(), name='news')
    assert 'news' in member_tool_description(agent, 'news')


def test_stringify_output_variants() -> None:
    assert stringify_output('plain') == 'plain'
    assert stringify_output(_SampleModel(value='x')) == '{"value":"x"}'
    assert stringify_output(123) == '123'


def test_sanitize_via_member_tool_name_special_chars() -> None:
    agent = Agent(TestModel(), name='!!!')
    assert member_tool_name(agent, 0) == 'member'


async def test_run_member_agent_and_team() -> None:
    agent = Agent(TestModel(custom_output_text='from-agent'), name='a')
    with agent.override(model=TestModel(custom_output_text='from-agent')):
        assert await run_member(agent, 'hi') == 'from-agent'

    class _StubTeam(BaseTeam[object]):
        name = 'stub'

        @property
        def members(self) -> list[Agent[object, object]]:
            return []

        async def run(self, user_prompt: str, *, usage: RunUsage | None = None) -> TeamResult[object]:
            return TeamResult(data=f'stub:{user_prompt}', usage=usage or RunUsage())

    stub = _StubTeam()
    assert await run_member(stub, 'q') == 'stub:q'
    usage = RunUsage(requests=1)
    assert await run_member(stub, 'q', usage=usage) == 'stub:q'
    with agent.override(model=TestModel(custom_output_text='with-usage')):
        assert await run_member(agent, 'hi', usage=usage) == 'with-usage'
