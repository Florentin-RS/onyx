import time
from unittest.mock import MagicMock, patch

import pytest

from onyx.configs.constants import DocumentSource
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.mindtickle.connector import MindtickleConnector
from onyx.connectors.models import Document, HierarchyNode
from tests.utils.secret_names import TestSecret

pytestmark = pytest.mark.secrets(
    TestSecret.MINDTICKLE_API_KEY,
    TestSecret.MINDTICKLE_SECRET_KEY,
    TestSecret.MINDTICKLE_LEARNING_SITE_URL,
)

# A small, stable hub on the test learning site. Scoping to it keeps the test fast.
_TEST_HUB_NAME = "BDR Battlecards"
_TARGET_ASSET_TITLE_FRAGMENT = "Open Reel"
# A one-module series whose module is an Articulate Rise course.
_TEST_SERIES_NAME = "BD BattleCards"


def _credentials(test_secrets: dict[TestSecret, str]) -> dict[str, str]:
    return {
        "mindtickle_api_key": test_secrets[TestSecret.MINDTICKLE_API_KEY],
        "mindtickle_secret_key": test_secrets[TestSecret.MINDTICKLE_SECRET_KEY],
        "mindtickle_learning_site_url": test_secrets[
            TestSecret.MINDTICKLE_LEARNING_SITE_URL
        ],
    }


@pytest.fixture
def mindtickle_connector(
    test_secrets: dict[TestSecret, str],
) -> MindtickleConnector:
    connector = MindtickleConnector(
        hub_names=[_TEST_HUB_NAME], index_training_modules=False, batch_size=5
    )
    connector.load_credentials(_credentials(test_secrets))
    return connector


def _collect_documents(connector: MindtickleConnector) -> list[Document]:
    documents: list[Document] = []
    for batch in connector.load_from_state():
        for item in batch:
            if isinstance(item, HierarchyNode):
                continue
            documents.append(item)
    return documents


@patch(
    "onyx.file_processing.extract_file_text.get_unstructured_api_key",
    return_value=None,
)
def test_mindtickle_load_from_state(
    mock_get_api_key: MagicMock,  # noqa: ARG001
    mindtickle_connector: MindtickleConnector,
) -> None:
    documents = _collect_documents(mindtickle_connector)

    assert len(documents) > 0
    for document in documents:
        assert document.source == DocumentSource.MINDTICKLE
        assert document.id.startswith("MINDTICKLE_ASSET_")
        assert document.semantic_identifier
        assert len(document.sections) == 1
        assert document.sections[0].text
        assert document.metadata["hubs"] == [_TEST_HUB_NAME]

    target = [
        document
        for document in documents
        if _TARGET_ASSET_TITLE_FRAGMENT in document.semantic_identifier
    ]
    assert target, "expected the Open Reel battle card in the test hub"
    section = target[0].sections[0]
    assert section.link is not None
    # The transcript of the PDF should be far longer than a title-only fallback.
    assert section.text is not None
    assert len(section.text) > 200
    assert target[0].doc_updated_at is not None


def test_mindtickle_slim_docs_cover_full_docs(
    mindtickle_connector: MindtickleConnector,
) -> None:
    with patch(
        "onyx.file_processing.extract_file_text.get_unstructured_api_key",
        return_value=None,
    ):
        full_ids = {
            document.id for document in _collect_documents(mindtickle_connector)
        }

    slim_ids: set[str] = set()
    for batch in mindtickle_connector.retrieve_all_slim_docs():
        slim_ids.update(
            item.id for item in batch if not isinstance(item, HierarchyNode)
        )

    assert slim_ids
    assert full_ids.issubset(slim_ids)


def test_mindtickle_poll_far_future_window_is_empty(
    mindtickle_connector: MindtickleConnector,
) -> None:
    start = time.time() + 365 * 24 * 60 * 60
    batches = list(mindtickle_connector.poll_source(start, start + 60))
    assert batches == []


def test_mindtickle_validate_settings(
    mindtickle_connector: MindtickleConnector,
) -> None:
    mindtickle_connector.validate_connector_settings()


def test_mindtickle_training_module_text(
    test_secrets: dict[TestSecret, str],
) -> None:
    connector = MindtickleConnector(
        index_asset_hub=False, series_names=[_TEST_SERIES_NAME]
    )
    connector.load_credentials(_credentials(test_secrets))

    documents = _collect_documents(connector)

    assert documents
    for document in documents:
        assert document.id.startswith("MINDTICKLE_MODULE_")
        assert document.metadata["kind"] == "training_module"
        assert document.metadata["series"] == [_TEST_SERIES_NAME]
        assert document.doc_updated_at is None
    battle_cards = [d for d in documents if "Battle Cards" in d.semantic_identifier]
    assert battle_cards, "expected the BD Battle Cards module"
    section = battle_cards[0].sections[0]
    assert section.link is not None and "jitredirectsso" in section.link
    # The Rise course text is far longer than the module description.
    assert section.text is not None and len(section.text) > 1000
    assert "OpenReel" in section.text


def test_mindtickle_validate_rejects_unknown_series(
    test_secrets: dict[TestSecret, str],
) -> None:
    connector = MindtickleConnector(
        index_asset_hub=False, series_names=["this series does not exist"]
    )
    connector.load_credentials(_credentials(test_secrets))
    with pytest.raises(ConnectorValidationError):
        connector.validate_connector_settings()


def test_mindtickle_validate_rejects_unknown_hub(
    test_secrets: dict[TestSecret, str],
) -> None:
    connector = MindtickleConnector(
        hub_names=["this hub does not exist"], index_training_modules=False
    )
    connector.load_credentials(_credentials(test_secrets))
    with pytest.raises(ConnectorValidationError):
        connector.validate_connector_settings()


def test_mindtickle_validate_rejects_unknown_learning_site(
    test_secrets: dict[TestSecret, str],
) -> None:
    connector = MindtickleConnector()
    credentials = _credentials(test_secrets)
    credentials["mindtickle_learning_site_url"] = "does-not-exist.mindtickle.com"
    connector.load_credentials(credentials)
    with pytest.raises(ConnectorValidationError):
        connector.validate_connector_settings()
