from __future__ import annotations

import pydantic_team


def test_package_version() -> None:
    assert pydantic_team.__version__ == '0.3.0'
