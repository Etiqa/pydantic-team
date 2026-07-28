from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_team import HierarchicalTeam, TeamResult


def _named_member(name: str, output: str) -> Agent[object, object]:
    return Agent(TestModel(custom_output_text=output), name=name, instructions=f'You are {name}')


def test_hierarchical_requires_leader() -> None:
    with pytest.raises(ValueError, match='leader_agent or leader_model'):
        HierarchicalTeam(members=[_named_member('a', 'x')])


def test_hierarchical_rejects_both_leader_sources() -> None:
    leader = Agent(TestModel(), name='leader')
    with pytest.raises(ValueError, match='not both'):
        HierarchicalTeam(
            leader_agent=leader,
            leader_model='test',
            members=[_named_member('a', 'x')],
        )


def test_hierarchical_requires_members() -> None:
    with pytest.raises(ValueError, match='members'):
        HierarchicalTeam(leader_model='test', members=[])


async def test_hierarchical_run_returns_team_result() -> None:
    leader = Agent(TestModel(), name='leader', instructions='Coordinate specialists.')
    member = _named_member('researcher', 'research-notes')
    team = HierarchicalTeam(leader_agent=leader, members=[member])

    with leader.override(model=TestModel(custom_output_text='final-answer')):
        with member.override(model=TestModel(custom_output_text='research-notes')):
            result = await team.run('Summarize AI news')

    assert isinstance(result, TeamResult)
    assert result.data == 'final-answer'
    assert result.usage.requests >= 1


async def test_hierarchical_registers_member_tools() -> None:
    leader_model = TestModel()
    leader = Agent(leader_model, name='leader')
    member = _named_member('analyst', 'analysis')
    team = HierarchicalTeam(leader_agent=leader, members=[member])

    with leader.override(model=leader_model):
        with member.override(model=TestModel(custom_output_text='analysis')):
            await team.run('Analyze this')

    assert leader_model.last_model_request_parameters is not None
    tool_names = [t.name for t in leader_model.last_model_request_parameters.function_tools]
    assert 'analyst' in tool_names


async def test_hierarchical_from_leader_model_string() -> None:
    member = _named_member('writer', 'draft')
    team = HierarchicalTeam(
        leader_model='test',
        members=[member],
        system_prompt_override='Delegate writing tasks.',
    )
    assert list(team.members) == [member]

    with team.leader.override(model=TestModel(custom_output_text='edited')):
        with member.override(model=TestModel(custom_output_text='draft')):
            result = await team.run('Write a blurb')
    assert result.data == 'edited'


async def test_hierarchical_run_with_shared_usage() -> None:
    from pydantic_ai.usage import RunUsage

    leader = Agent(TestModel(), name='leader')
    member = _named_member('worker', 'done')
    team = HierarchicalTeam(leader_agent=leader, members=[member])
    usage = RunUsage()

    with leader.override(model=TestModel(custom_output_text='ok')):
        with member.override(model=TestModel(custom_output_text='done')):
            result = await team.run('task', usage=usage)

    assert result.data == 'ok'
    assert result.usage.requests >= 1


async def test_hierarchical_system_prompt_override_on_existing_leader() -> None:
    leader = Agent(TestModel(), name='leader', instructions='Base instructions')
    member = _named_member('worker', 'done')
    team = HierarchicalTeam(
        leader_agent=leader,
        members=[member],
        system_prompt_override='Always delegate to worker.',
    )
    with leader.override(model=TestModel(custom_output_text='ok')):
        with member.override(model=TestModel(custom_output_text='done')):
            result = await team.run('Do the thing')
    assert result.data == 'ok'


async def test_hierarchical_default_leader_instructions() -> None:
    member = _named_member('solo', 'out')
    team = HierarchicalTeam(leader_model='test', members=[member])
    with team.leader.override(model=TestModel(custom_output_text='final')):
        with member.override(model=TestModel(custom_output_text='out')):
            result = await team.run('hello')
    assert result.data == 'final'


async def test_hierarchical_unnamed_member_tool() -> None:
    leader_model = TestModel()
    leader = Agent(leader_model, name='leader')
    member = Agent(TestModel(custom_output_text='x'))
    team = HierarchicalTeam(leader_agent=leader, members=[member])

    with leader.override(model=leader_model):
        with member.override(model=TestModel(custom_output_text='x')):
            await team.run('go')

    assert leader_model.last_model_request_parameters is not None
    tool_names = [t.name for t in leader_model.last_model_request_parameters.function_tools]
    assert 'member_0' in tool_names

    inner_member = _named_member('inner', 'inner-out')
    nested = HierarchicalTeam(
        name='inner_team',
        leader_model='test',
        members=[inner_member],
        system_prompt_override='Inner leader',
    )
    outer_leader = Agent(TestModel(), name='outer_leader')
    outer = HierarchicalTeam(leader_agent=outer_leader, members=[nested])

    with outer_leader.override(model=TestModel(custom_output_text='outer-final')):
        with nested.leader.override(model=TestModel(custom_output_text='nested-final')):
            with inner_member.override(model=TestModel(custom_output_text='inner-out')):
                result = await outer.run('Top request')

    assert result.data == 'outer-final'
    assert 'inner_team' == nested.name
