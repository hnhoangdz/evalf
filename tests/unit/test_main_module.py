import runpy

import pytest

pytestmark = pytest.mark.unit


def test_package_main_delegates_to_cli(monkeypatch) -> None:
    called = {}

    def fake_main() -> None:
        called["invoked"] = True

    monkeypatch.setattr("evalf.cli.main", fake_main)

    runpy.run_module("evalf.__main__", run_name="__main__")

    assert called["invoked"] is True
