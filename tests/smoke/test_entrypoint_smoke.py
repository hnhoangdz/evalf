import runpy

import pytest

pytestmark = pytest.mark.smoke


def test_package_entrypoint_smoke(monkeypatch) -> None:
    called = {}

    def fake_main() -> None:
        called["invoked"] = True

    monkeypatch.setattr("evalf.cli.main", fake_main)

    runpy.run_module("evalf.__main__", run_name="__main__")

    assert called["invoked"] is True
