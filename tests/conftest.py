import pytest

from rmf_agent.config import Settings
from rmf_agent.tools import Toolbox, build_toolbox
from rmf_agent.tools.audit import AuditLog


@pytest.fixture
def toolbox_for():
    """Build an in-process toolbox for a scenario; audit records stay in memory."""

    def make(scenario: str = "nominal", **settings) -> Toolbox:
        config = Settings(scenario=scenario, audit_path=None, **settings)
        return build_toolbox(config, audit=AuditLog(None))

    return make
