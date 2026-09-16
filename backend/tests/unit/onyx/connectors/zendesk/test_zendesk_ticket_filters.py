from datetime import datetime, timezone

import pytest

from onyx.connectors.zendesk.connector import _parse_created_after
from onyx.connectors.zendesk.connector import ZendeskConnector


def _ticket(created_at: str, status: str = "open") -> dict:
    return {"id": 1, "created_at": created_at, "status": status}


def test_parse_created_after_accepts_date_and_datetime() -> None:
    assert _parse_created_after(None) is None
    assert _parse_created_after("  ") is None
    assert _parse_created_after("2026-09-01") == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert _parse_created_after("2026-09-01T12:30:00Z") == datetime(
        2026, 9, 1, 12, 30, tzinfo=timezone.utc
    )


def test_no_filters_keeps_everything_but_deleted() -> None:
    c = ZendeskConnector(content_type="tickets")
    assert not c._should_skip_ticket(_ticket("2024-01-01T00:00:00Z", "closed"))
    assert c._should_skip_ticket(_ticket("2026-09-10T00:00:00Z", "deleted"))


def test_created_after_skips_old_tickets_even_when_recently_updated() -> None:
    c = ZendeskConnector(content_type="tickets", tickets_created_after="2026-09-01")
    old = _ticket("2025-07-17T09:00:00Z")
    old["updated_at"] = "2026-09-10T23:28:37Z"  # bulk automation touched it
    assert c._should_skip_ticket(old)
    assert not c._should_skip_ticket(_ticket("2026-09-01T00:00:00Z"))
    assert not c._should_skip_ticket(_ticket("2026-09-14T18:07:09Z"))


def test_created_after_keeps_ticket_with_missing_or_bad_created_at() -> None:
    c = ZendeskConnector(content_type="tickets", tickets_created_after="2026-09-01")
    assert not c._should_skip_ticket({"id": 1, "status": "open"})
    assert not c._should_skip_ticket(_ticket("not-a-date"))


def test_exclude_statuses_is_case_insensitive() -> None:
    c = ZendeskConnector(
        content_type="tickets", exclude_ticket_statuses=["Closed", " solved "]
    )
    assert c._should_skip_ticket(_ticket("2026-09-10T00:00:00Z", "closed"))
    assert c._should_skip_ticket(_ticket("2026-09-10T00:00:00Z", "SOLVED"))
    assert not c._should_skip_ticket(_ticket("2026-09-10T00:00:00Z", "open"))


@pytest.mark.parametrize("bad", ["2026/09/01", "yesterday"])
def test_parse_created_after_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_created_after(bad)
