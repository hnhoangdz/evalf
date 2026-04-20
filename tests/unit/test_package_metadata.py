from importlib.metadata import version

import pytest

import evalf

pytestmark = pytest.mark.unit


def test_package_version_matches_installed_metadata() -> None:
    assert evalf.__version__ == version("evalf")
