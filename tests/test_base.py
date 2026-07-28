from __future__ import annotations

from pydantic_ai.usage import RunUsage

from pydantic_team.base import TeamResult


def test_team_result_holds_data_and_usage() -> None:
    usage = RunUsage(input_tokens=3, output_tokens=5, requests=1)
    result = TeamResult(data={'answer': 42}, usage=usage)
    assert result.data == {'answer': 42}
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 5
    assert result.usage.requests == 1


def test_team_result_allows_string_data() -> None:
    result = TeamResult(data='done', usage=RunUsage())
    assert result.data == 'done'
