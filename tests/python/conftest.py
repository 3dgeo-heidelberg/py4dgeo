from py4dgeo.util import MemoryPolicy, set_memory_policy

import pytest


@pytest.fixture(autouse=True)
def memory_policy_fixture():
    """This fixture ensures that all tests start with the default memory policy"""
    set_memory_policy(MemoryPolicy.COREPOINTS)
