import tomllib
from pathlib import Path

import crypto_realtime_pipeline


def test_version_matches_project_metadata():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

    assert crypto_realtime_pipeline.__version__ == expected
