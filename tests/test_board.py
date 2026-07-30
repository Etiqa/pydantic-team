from __future__ import annotations

import asyncio

import pytest

from pydantic_team.board import TaskBoard, TaskClaimError, TaskNotFoundError, TaskStatus


async def test_post_and_list_messages() -> None:
    board = TaskBoard()
    msg = await board.post_message('alice', 'bob', 'need sources')
    assert msg.id.startswith('msg-')
    assert msg.sender == 'alice'
    assert msg.to == 'bob'
    assert msg.body == 'need sources'
    assert msg.task_id is None
    assert board.messages_snapshot() == [msg]
    assert await board.list_messages() == [msg]


async def test_list_messages_filters_for_agent() -> None:
    board = TaskBoard()
    dm = await board.post_message('alice', 'bob', 'private')
    broadcast = await board.post_message('alice', '*', 'hello all')
    other = await board.post_message('carol', 'dave', 'ignore')
    _ = other
    visible = await board.list_messages(agent_id='bob')
    assert {m.id for m in visible} == {dm.id, broadcast.id}
    # Sender sees their own outbound messages.
    alice_view = await board.list_messages(agent_id='alice')
    assert {m.id for m in alice_view} == {dm.id, broadcast.id}


async def test_post_message_with_task_id() -> None:
    board = TaskBoard()
    task = await board.add_task('Research')
    msg = await board.post_message('alice', 'bob', 'about this', task_id=task.id)
    assert msg.task_id == task.id


async def test_post_message_unknown_task_fails() -> None:
    board = TaskBoard()
    with pytest.raises(TaskNotFoundError):
        await board.post_message('alice', 'bob', 'x', task_id='missing')


async def test_post_message_direct_signals_wakeup_recipient() -> None:
    board = TaskBoard()
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    await board.post_message('alice', 'bob', 'ping')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == {'bob'}


async def test_post_message_broadcast_signals_wakeup_without_agent() -> None:
    board = TaskBoard()
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    await board.post_message('alice', '*', 'ping all')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == set()


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


async def test_complete_without_reviewer_goes_done() -> None:
    board = TaskBoard()
    task = await board.add_task('Plain')
    await board.claim(task.id, 'alice')
    done = await board.complete(task.id, result='ok', agent_id='alice')
    assert done.status is TaskStatus.DONE
    assert done.result == 'ok'
    assert board.is_complete()


async def test_complete_with_reviewer_goes_pending_review() -> None:
    board = TaskBoard()
    task = await board.add_task('Gated', reviewer='leader')
    assert task.reviewer == 'leader'
    await board.claim(task.id, 'alice')
    pending = await board.complete(task.id, result='draft', agent_id='alice')
    assert pending.status is TaskStatus.PENDING_REVIEW
    assert pending.result == 'draft'
    assert pending.rejection_reason is None
    assert not board.is_complete()


async def test_complete_with_reviewer_wakes_reviewer() -> None:
    board = TaskBoard()
    task = await board.add_task('Gated', reviewer='leader')
    await board.claim(task.id, 'alice')
    await board.wait_wakeup()
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    await board.complete(task.id, result='draft', agent_id='alice')
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == {'leader'}


async def test_approve_and_reject_review_flow() -> None:
    board = TaskBoard()
    task = await board.add_task('Revise me', reviewer='leader')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='v1', agent_id='alice')
    rejected = await board.reject(task.id, reason='too short', agent_id='leader')
    assert rejected.status is TaskStatus.NEEDS_REVISION
    assert rejected.result == 'v1'
    assert rejected.rejection_reason == 'too short'
    revised = await board.complete(task.id, result='v2 longer', agent_id='alice')
    assert revised.status is TaskStatus.PENDING_REVIEW
    assert revised.result == 'v2 longer'
    assert revised.rejection_reason is None
    with pytest.raises(TaskClaimError):
        await board.approve(task.id, agent_id='alice')
    approved = await board.approve(task.id, agent_id='leader')
    assert approved.status is TaskStatus.DONE
    assert board.is_complete()


async def test_reject_requires_non_empty_reason() -> None:
    board = TaskBoard()
    task = await board.add_task('X', reviewer='leader')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='v1', agent_id='alice')
    with pytest.raises(TaskClaimError, match='reason'):
        await board.reject(task.id, reason='  ', agent_id='leader')


async def test_reject_wrong_reviewer_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('X', reviewer='leader')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='v1', agent_id='alice')
    with pytest.raises(TaskClaimError):
        await board.reject(task.id, reason='nope', agent_id='bob')


async def test_approve_wrong_status_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('X', reviewer='leader')
    await board.claim(task.id, 'alice')
    with pytest.raises(TaskClaimError):
        await board.approve(task.id, agent_id='leader')


async def test_assign_pending_review_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('X', reviewer='leader')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='v1', agent_id='alice')
    with pytest.raises(TaskClaimError, match='pending review'):
        await board.assign(task.id, 'bob')


async def test_assign_reviewer_and_wakeup_when_pending() -> None:
    board = TaskBoard()
    task = await board.add_task('X', reviewer='leader')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='v1', agent_id='alice')
    await board.wait_wakeup()
    waiter = asyncio.create_task(board.wait_wakeup())
    await asyncio.sleep(0)
    updated = await board.assign_reviewer(task.id, 'verifier')
    assert updated.reviewer == 'verifier'
    agents = await asyncio.wait_for(waiter, timeout=1)
    assert agents == {'verifier'}


async def test_reject_from_done_not_allowed() -> None:
    board = TaskBoard()
    task = await board.add_task('X')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='ok', agent_id='alice')
    with pytest.raises(TaskClaimError):
        await board.reject(task.id, reason='too late', agent_id='leader')


async def test_assign_reviewer_on_claimed_does_not_require_pending() -> None:
    board = TaskBoard()
    task = await board.add_task('X')
    await board.claim(task.id, 'alice')
    updated = await board.assign_reviewer(task.id, 'leader')
    assert updated.reviewer == 'leader'
    assert updated.status is TaskStatus.CLAIMED
    assert board.snapshot()[0].reviewer == 'leader'


async def test_assign_reviewer_on_done_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('X')
    await board.claim(task.id, 'alice')
    await board.complete(task.id, result='ok', agent_id='alice')
    with pytest.raises(TaskClaimError):
        await board.assign_reviewer(task.id, 'leader')


async def test_complete_open_task_fails() -> None:
    board = TaskBoard()
    task = await board.add_task('X')
    with pytest.raises(TaskClaimError, match='not completable'):
        await board.complete(task.id, result='x', agent_id='alice')
