import pathlib

import yaml


def test_ci_workflow():
    p = pathlib.Path(".github/workflows/ci.yml")
    assert p.exists()
    assert yaml.safe_load(p.read_text())
    assert "ruff" in p.read_text() and "pytest" in p.read_text()
