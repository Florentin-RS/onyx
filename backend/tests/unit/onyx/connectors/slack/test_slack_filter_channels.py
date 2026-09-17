"""Channel include/exclude filtering for the Slack connector: excludes apply
after includes, support exact-name and full-match regex modes, and never fail
on names absent from the workspace."""

from typing import Any, cast

import pytest

from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.slack.connector import _validate_channel_regexes, filter_channels
from onyx.connectors.slack.models import ChannelType


def _channel(name: str, is_member: bool = True) -> ChannelType:
    base: dict[str, Any] = {"id": f"C-{name}", "name": name, "is_member": is_member}
    return cast(ChannelType, base)


CHANNELS = [
    _channel("general"),
    _channel("support"),
    _channel("infra-alerts", is_member=False),
    _channel("billing-alerts"),
    _channel("deploy-notifications", is_member=False),
]


def _names(channels: list[ChannelType]) -> list[str]:
    return [channel["name"] for channel in channels]


def test_no_filters_returns_all() -> None:
    assert filter_channels(CHANNELS, None, False) == CHANNELS


def test_exclude_regex_without_includes() -> None:
    result = filter_channels(
        CHANNELS,
        None,
        False,
        channels_to_exclude=[".*-alerts", ".*-notifications"],
        exclude_regex_enabled=True,
    )
    assert _names(result) == ["general", "support"]


def test_exclude_exact_names() -> None:
    result = filter_channels(
        CHANNELS,
        None,
        False,
        channels_to_exclude=["support"],
        exclude_regex_enabled=False,
    )
    assert _names(result) == [
        "general",
        "infra-alerts",
        "billing-alerts",
        "deploy-notifications",
    ]


def test_exclude_exact_requires_full_name() -> None:
    result = filter_channels(
        CHANNELS,
        None,
        False,
        channels_to_exclude=["alerts"],
        exclude_regex_enabled=False,
    )
    assert result == CHANNELS


def test_exclude_applied_after_include_regex() -> None:
    result = filter_channels(
        CHANNELS,
        ["infra-.*", "general"],
        True,
        channels_to_exclude=[".*-alerts"],
        exclude_regex_enabled=True,
    )
    assert _names(result) == ["general"]


def test_exclude_applied_after_include_exact() -> None:
    result = filter_channels(
        CHANNELS,
        ["general", "support"],
        False,
        channels_to_exclude=["support"],
        exclude_regex_enabled=False,
    )
    assert _names(result) == ["general"]


def test_exclude_nonexistent_name_is_harmless() -> None:
    result = filter_channels(
        CHANNELS,
        None,
        False,
        channels_to_exclude=["no-such-channel"],
        exclude_regex_enabled=False,
    )
    assert result == CHANNELS


def test_exclude_regex_is_full_match() -> None:
    # "alerts" only partially matches "infra-alerts", so nothing is excluded
    result = filter_channels(
        CHANNELS,
        None,
        False,
        channels_to_exclude=["alerts"],
        exclude_regex_enabled=True,
    )
    assert result == CHANNELS


def test_include_validation_still_raises_for_unknown_channel() -> None:
    with pytest.raises(ValueError, match="not found in workspace"):
        filter_channels(CHANNELS, ["typo-channel"], False)


def test_member_channels_only_keeps_joined_channels() -> None:
    result = filter_channels(CHANNELS, None, False, member_channels_only=True)
    assert _names(result) == ["general", "support", "billing-alerts"]


def test_member_channels_only_is_off_by_default() -> None:
    assert filter_channels(CHANNELS, None, False) == CHANNELS


def test_member_channels_only_combines_with_include_regex() -> None:
    result = filter_channels(CHANNELS, [".*-alerts"], True, member_channels_only=True)
    assert _names(result) == ["billing-alerts"]


def test_member_channels_only_combines_with_exclude() -> None:
    result = filter_channels(
        CHANNELS,
        None,
        False,
        channels_to_exclude=["support"],
        exclude_regex_enabled=False,
        member_channels_only=True,
    )
    assert _names(result) == ["general", "billing-alerts"]


def test_member_channels_only_rejects_included_channel_bot_is_not_in() -> None:
    with pytest.raises(
        ValueError, match="not found among channels the bot is a member of"
    ):
        filter_channels(CHANNELS, ["infra-alerts"], False, member_channels_only=True)


def test_validate_channel_regexes_accepts_valid_patterns() -> None:
    _validate_channel_regexes([".*-alerts$", "general"], "channel")


def test_validate_channel_regexes_rejects_invalid_pattern() -> None:
    with pytest.raises(ConnectorValidationError, match="Invalid channel regex"):
        _validate_channel_regexes(["[unclosed"], "channel")


def test_validate_channel_regexes_handles_none() -> None:
    _validate_channel_regexes(None, "channel")
