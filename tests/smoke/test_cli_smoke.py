import pytest
from click.testing import CliRunner

from evalf.cli import cli

pytestmark = pytest.mark.smoke


def test_list_metrics_smoke() -> None:
    result = CliRunner().invoke(cli, ["list-metrics"])

    assert result.exit_code == 0
    assert "faithfulness" in result.output
    assert "c4" in result.output
