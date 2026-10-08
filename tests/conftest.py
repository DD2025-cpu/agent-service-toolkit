import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Keep OS home discovery working while removing application settings."""
    os_home = {
        name: os.environ[name]
        for name in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
        if name in os.environ
    }
    with patch.dict(os.environ, os_home, clear=True):
        yield
