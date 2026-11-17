import pathlib

import pytest


@pytest.fixture
def fixture_raw():
    return pathlib.Path(__file__).parent / "fixtures" / "ufcstats"
