from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pydantic_team import CollaborativeTeam, TeamResult
from pydantic_team.board import TaskBoard
from pydantic_team.collaborative import (
    BoardDeps,
    add_task,
    assign_task,
    claim_task,
    complete_task,
    default_leader_instructions,
    list_tasks,
    member_work_prompt,
    seed_user_prompt,
)


@dataclass
class _FakeCtx:
    deps: object


def _ctx(board: TaskBoard, agent_id: str) -> RunContext[object]:
    return cast(RunContext[object], _FakeCtx(deps=BoardDeps(board=board, agent_id=agent_id)))


def test_collaborative_requires_leader() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='leader_agent or leader_model'):
        CollaborativeTeam(members=[member])


def test_collaborative_rejects_both_leader_sources() -> None:
    leader = Agent(TestModel(), name='leader')
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='not both'):
        CollaborativeTeam(leader_agent=leader, leader_model='test', members=[member])


def test_collaborative_requires_members() -> None:
    with pytest.raises(ValueError, match='members'):
        CollaborativeTeam(leader_model='test', members=[])


def test_collaborative_requires_positive_rounds() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='max_rounds'):
        CollaborativeTeam(leader_model='test', members=[member], max_rounds=0)


async def test_collaborative_run_returns_team_result() -> None:
    leader = Agent(TestModel(), name='leader', instructions='Lead the board.')
    worker = Agent(TestModel(), name='worker', instructions='Claim and complete tasks.')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=2)
    assert list(team.members) == [worker]
    assert team.leader is leader

    with leader.override(model=TestModel(custom_output_text='final-summary')):
        with worker.override(model=TestModel(custom_output_text='worker-done')):
            result = await team.run('Ship the feature')

    assert isinstance(result, TeamResult)
    assert result.data == 'final-summary'
    assert result.usage.requests >= 1


async def test_collaborative_no_tasks_returns_lead_output() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=2)

    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo-lead')):
        with worker.override(model=TestModel(custom_output_text='should-not-matter')):
            result = await team.run('Nothing to split')

    assert result.data == 'solo-lead'


async def test_collaborative_leader_tools_create_tasks() -> None:
    leader_model = TestModel()
    leader = Agent(leader_model, name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=leader_model):
        with worker.override(model=TestModel(custom_output_text='ok')):
            await team.run('Create work items')

    assert leader_model.last_model_request_parameters is not None
    tool_names = {t.name for t in leader_model.last_model_request_parameters.function_tools}
    assert 'add_task' in tool_names
    assert 'assign_task' in tool_names
    assert 'list_tasks' in tool_names


async def test_collaborative_member_tools_include_claim() -> None:
    leader = Agent(TestModel(), name='leader')
    worker_model = TestModel()
    worker = Agent(worker_model, name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=TestModel(custom_output_text='lead')):
        with worker.override(model=worker_model):
            await team.run('Do parallel work')

    assert worker_model.last_model_request_parameters is not None
    tool_names = {t.name for t in worker_model.last_model_request_parameters.function_tools}
    assert 'claim_task' in tool_names
    assert 'complete_task' in tool_names


async def test_collaborative_from_leader_model() -> None:
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_model='test',
        members=[worker],
        system_prompt_override='Seed tasks then synthesize.',
        max_rounds=1,
    )
    with team.leader.override(model=TestModel(custom_output_text='done')):
        with worker.override(model=TestModel(custom_output_text='w')):
            result = await team.run('Goal')
    assert result.data == 'done'


async def test_collaborative_system_prompt_on_existing_leader() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        system_prompt_override='Always create at least one task.',
        max_rounds=1,
    )
    with leader.override(model=TestModel(custom_output_text='ok')):
        with worker.override(model=TestModel(custom_output_text='ok')):
            result = await team.run('x')
    assert result.data == 'ok'


async def test_collaborative_unnamed_agents_use_fallback_ids() -> None:
    leader = Agent(TestModel())
    worker = Agent(TestModel())
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)
    with leader.override(model=TestModel(call_tools=[], custom_output_text='ok')):
        with worker.override(model=TestModel(custom_output_text='ok')):
            result = await team.run('x')
    assert result.data == 'ok'


async def test_board_tools_error_and_success_paths() -> None:
    board = TaskBoard()
    ctx = _ctx(board, 'alice')

    assert await list_tasks(ctx, None) == 'No tasks'
    assert 'Error: invalid status' in await list_tasks(ctx, 'nope')
    created_msg = await add_task(ctx, 'Research notes')
    assert 'Created' in created_msg

    created = (await board.list_tasks())[0]
    listed = await list_tasks(ctx, 'open')
    assert created.id in listed

    assert 'Error' in await assign_task(ctx, 'missing', 'alice')
    assert 'Assigned' in await assign_task(ctx, created.id, 'alice')

    board2 = TaskBoard()
    open_task = await board2.add_task('U')
    ctx2 = _ctx(board2, 'bob')
    assert 'Claimed' in await claim_task(ctx2, open_task.id)
    assert 'Error' in await claim_task(ctx2, open_task.id)
    assert 'Completed' in await complete_task(ctx2, open_task.id, 'done')
    assert 'Error' in await complete_task(ctx2, open_task.id, 'again')


async def test_board_deps_type_error() -> None:
    with pytest.raises(TypeError, match='BoardDeps'):
        await list_tasks(cast(RunContext[object], _FakeCtx(deps='nope')), None)


def test_default_leader_instructions_require_assign_by_role() -> None:
    text = default_leader_instructions(['researcher', 'writer'])
    assert 'researcher' in text
    assert 'writer' in text
    assert 'assign_task' in text
    assert 'Do not leave tasks open without an assignee' in text


def test_seed_user_prompt_includes_roster() -> None:
    text = seed_user_prompt('Ship it', ['researcher', 'writer'])
    assert 'Ship it' in text
    assert 'researcher' in text
    assert 'assign_task' in text


async def test_member_work_prompt_lists_only_own_assignments() -> None:
    board = TaskBoard()
    research = await board.add_task('Research notes')
    writing = await board.add_task('Draft copy')
    await board.assign(research.id, 'researcher')
    await board.assign(writing.id, 'writer')

    researcher_prompt = member_work_prompt('Brief', 'researcher', board)
    assert "Your agent id is 'researcher'" in researcher_prompt
    assert 'do not claim tasks assigned to others' in researcher_prompt
    assert research.id in researcher_prompt
    assert 'Tasks assigned to you:\n- task-1' in researcher_prompt

    writer_prompt = member_work_prompt('Brief', 'writer', board)
    assert writing.id in writer_prompt
    assert "Your agent id is 'writer'" in writer_prompt


async def test_member_work_prompt_none_when_unassigned() -> None:
    board = TaskBoard()
    await board.add_task('Open work')
    prompt = member_work_prompt('Goal', 'researcher', board)
    assert 'Tasks assigned to you:\n(none)' in prompt


async def test_collaborative_from_leader_model_includes_member_roster() -> None:
    researcher = Agent(TestModel(), name='researcher')
    writer = Agent(TestModel(), name='writer')
    team = CollaborativeTeam(leader_model='test', members=[researcher, writer], max_rounds=1)
    assert 'researcher' in team.member_ids
    assert 'writer' in team.member_ids
    instructions = default_leader_instructions(list(team.member_ids))
    assert 'researcher' in instructions
    assert 'assign_task' in instructions
