# content of conftest.py

import pytest
import asyncio


def pytest_addoption(parser):
    parser.addoption(
        "--e2e", action="store_true", default=False, help="run e2e e2e_tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--e2e"):
        # --e2e given in cli: do not skip e2e e2e_tests
        return
    skip_e2e = pytest.mark.skip(reason="need --e2e option to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


# override the scope of the event loop, by default it is function scoped
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()
