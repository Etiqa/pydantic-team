# API reference

Public exports from [`pydantic_team`][pydantic_team].

::: pydantic_team
    options:
      members:
        - BaseTeam
        - TeamResult
        - HierarchicalTeam
        - CollaborativeTeam
        - CollaborativeRun
        - TeamTask
        - TasksScheduled
        - TaskCompleted
        - PhaseJoined
        - MessagePosted
        - TaskReviewDecided
        - RunEnded
        - TaskBoard
        - Task
        - BoardMessage
        - TaskStatus
        - BoardDeps
        - instrument_pydantic_team
        - is_instrumented
        - __version__

## Base types

::: pydantic_team.base.TeamResult

::: pydantic_team.base.BaseTeam

## Hierarchical team

::: pydantic_team.hierarchical.HierarchicalTeam

## Collaborative team / board

::: pydantic_team.board.TaskStatus

::: pydantic_team.board.Task

::: pydantic_team.board.BoardMessage

::: pydantic_team.board.TaskBoard

::: pydantic_team.collaborative.BoardDeps

::: pydantic_team.collaborative.CollaborativeTeam

::: pydantic_team.collaborative.CollaborativeRun

## Run events

::: pydantic_team.events.TeamTask

::: pydantic_team.events.TasksScheduled

::: pydantic_team.events.TaskCompleted

::: pydantic_team.events.PhaseJoined

::: pydantic_team.events.MessagePosted

::: pydantic_team.events.TaskReviewDecided

::: pydantic_team.events.RunEnded
