from __future__ import annotations

import asyncio

import pytest

from pydantic_team.board import TaskBoard, TaskClaimError, TaskNotFoundError, TaskStatus


async def test_add_and_list_tasks() -> None:
    board = TaskBoard()
    t1 = await board.add_task('Research', 'Find sources')
    t2 = await board.add_task('Write', description='Draft')
    assert t1.id != t2.id
    assert t1.status is TaskStatus.OPEN
    tasks = await board.list_tasks()
    assert {t.id for t in tasks} == {t1.id, t2.id}


async def test_list_tasks_filters_by_status() -> None:
    board = TaskBoard()
    open_task = await board.add_task('A')
    claimed = await board.add_task('B')
    await board.claim(claimed.id, 'agent-1')
    assert [t.id for t in await board.list_tasks(status=TaskStatus.OPEN)] == [open_task.id]
    assert [t.id for t in await board.list_tasks(status=TaskStatus.CLAIMED)] == [claimed.id]


async def test_claim_then_second_claim_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('Solo')
    claimed = await board.claim(task.id, 'alice')
    assert claimed.status is TaskStatus.CLAIMED
    assert claimed.assignee == 'alice'
    with pytest.raises(TaskClaimError):
        await board.claim(task.id, 'bob')


async def test_assign_forces_assignee() -> None:
    board = TaskBoard()
    task = await board.add_task('Assigned')
    assigned = await board.assign(task.id, 'carol')
    assert assigned.status is TaskStatus.CLAIMED
    assert assigned.assignee == 'carol'
    reassigned = await board.assign(task.id, 'dave')
    assert reassigned.assignee == 'dave'


async def test_complete_task() -> None:
    board = TaskBoard()
    task = await board.add_task('Do it')
    await board.claim(task.id, 'alice')
    done = await board.complete(task.id, result='finished', agent_id='alice')
    assert done.status is TaskStatus.DONE
    assert done.result == 'finished'
    assert board.is_complete()


async def test_complete_wrong_assignee_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('Do it')
    await board.claim(task.id, 'alice')
    with pytest.raises(TaskClaimError):
        await board.complete(task.id, result='x', agent_id='bob')


async def test_claim_unknown_task_fails() -> None:
    board = TaskBoard()
    with pytest.raises(TaskNotFoundError):
        await board.claim('missing', 'alice')


async def test_parallel_claim_only_one_wins() -> None:
    board = TaskBoard()
    task = await board.add_task('Race')

    async def try_claim(agent_id: str) -> str | None:
        try:
            claimed = await board.claim(task.id, agent_id)
            return claimed.assignee
        except TaskClaimError:
            return None

    results = await asyncio.gather(try_claim('a'), try_claim('b'), try_claim('c'))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert winners[0] in {'a', 'b', 'c'}


async def test_empty_board_is_complete() -> None:
    assert TaskBoard().is_complete()


async def test_assign_done_task_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('X')
    await board.claim(task.id, 'a')
    await board.complete(task.id, result='ok', agent_id='a')
    with pytest.raises(TaskClaimError):
        await board.assign(task.id, 'b')


async def test_assign_signals_wakeup_with_assignee() -> None:
    board = TaskBoard()
    task = await board.add_task('Wake me')
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    await board.assign(task.id, 'researcher')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == {'researcher'}


async def test_claim_signals_wakeup_with_claimant() -> None:
    board = TaskBoard()
    task = await board.add_task('Claim me')
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    await board.claim(task.id, 'writer')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == {'writer'}


async def test_complete_signals_wakeup_without_agent() -> None:
    board = TaskBoard()
    task = await board.add_task('Finish me')
    await board.claim(task.id, 'alice')
    # Drain claim wakeup so complete's signal is observed alone.
    await board.wait_wakeup()
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    await board.complete(task.id, result='done', agent_id='alice')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == set()


async def test_signal_wakeup_wakes_waiter() -> None:
    board = TaskBoard()
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    board.signal_wakeup('solo')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == {'solo'}
