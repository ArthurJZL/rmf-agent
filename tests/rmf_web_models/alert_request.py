"""Fields copied from rmf-web api_server/models/alerts.py (AlertRequest) at the commit above."""

from enum import Enum

from pydantic import BaseModel


class AlertParameter(BaseModel):
    name: str
    value: str


class AlertRequest(BaseModel):
    class Tier(str, Enum):
        Info = "info"
        Warning = "warning"
        Error = "error"

    id: str
    unix_millis_alert_time: int
    title: str
    subtitle: str
    message: str
    display: bool
    tier: Tier
    responses_available: list[str]
    alert_parameters: list[AlertParameter]
    task_id: str | None
