import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from onyx.configs.constants import DocumentSource
from onyx.connectors.interfaces import GenerateDocumentsOutput
from onyx.connectors.mindtickle.client import (
    MindtickleClientError,
    normalize_learning_site_url,
)
from onyx.connectors.mindtickle.connector import MindtickleConnector
from onyx.connectors.mindtickle.models import (
    MindtickleAssetDetails,
    MindtickleAssetMedia,
    MindtickleAssetSummary,
    MindtickleHub,
)
from onyx.connectors.mindtickle.utils import (
    asset_type_to_extension,
    filter_hubs,
    transcript_to_text,
)
from onyx.connectors.models import Document, SlimDocument


def test_normalize_learning_site_url() -> None:
    assert (
        normalize_learning_site_url("https://RiversideU.mindtickle.com/")
        == "riversideu.mindtickle.com"
    )
    assert normalize_learning_site_url("acme.mindtickle.com") == "acme.mindtickle.com"


def test_transcript_plain_text_passthrough() -> None:
    assert transcript_to_text(b"  Battle Card\n\nPost Production  ") == (
        "Battle Card\n\nPost Production"
    )


def test_transcript_json_word_list() -> None:
    payload = {
        "results": {
            "words": [
                {"word": "All", "start_time": 0.0, "end_time": 0.1},
                {"word": "right", "start_time": 0.1, "end_time": 0.3},
                {"word": "team,", "start_time": 0.3, "end_time": 0.6},
            ]
        }
    }
    assert transcript_to_text(json.dumps(payload).encode()) == "All right team,"


def test_transcript_unknown_json_yields_empty() -> None:
    assert transcript_to_text(b'{"unexpected": 1}') == ""


def test_asset_type_to_extension() -> None:
    assert asset_type_to_extension("MEDIA_TYPE_DOCUMENT_PDF") == ".pdf"
    assert asset_type_to_extension("MEDIA_TYPE_DOCUMENT_WORD") == ".docx"
    assert asset_type_to_extension("PPT") == ".pptx"
    assert asset_type_to_extension("MEDIA_TYPE_IFRAME") is None
    assert asset_type_to_extension(None) is None


def test_filter_hubs_allow_and_exclude() -> None:
    hubs = [
        MindtickleHub(id="1", title="Competitive Intel"),
        MindtickleHub(id="2", title="Legal Resources"),
        MindtickleHub(id="3", title="Pricing"),
    ]
    assert [hub.id for hub in filter_hubs(hubs, None, None)] == ["1", "2", "3"]
    assert [hub.id for hub in filter_hubs(hubs, ["competitive intel "], None)] == ["1"]
    assert [hub.id for hub in filter_hubs(hubs, None, ["LEGAL RESOURCES"])] == [
        "1",
        "3",
    ]


def test_asset_summary_accepts_documented_and_live_keys() -> None:
    live = MindtickleAssetSummary.model_validate(
        {"id": "1", "title": "Live", "last_updated_time": "1789553966"}
    )
    documented = MindtickleAssetSummary.model_validate({"id": "2", "name": "Docs"})
    assert live.title == "Live"
    assert live.last_updated_time == 1789553966
    assert documented.title == "Docs"


def _documents(output: GenerateDocumentsOutput) -> list[Document]:
    return [doc for batch in output for doc in batch if isinstance(doc, Document)]


def _connector_with_mock_client() -> tuple[MindtickleConnector, MagicMock]:
    connector = MindtickleConnector()
    connector.load_credentials(
        {
            "mindtickle_api_key": "key",
            "mindtickle_secret_key": "secret",
            "mindtickle_learning_site_url": "acme.mindtickle.com",
        }
    )
    client = MagicMock()
    connector._client = client
    return connector, client


def test_load_from_state_dedupes_assets_across_hubs() -> None:
    connector, client = _connector_with_mock_client()
    client.list_hubs.return_value = [
        MindtickleHub(id="h1", title="Hub One"),
        MindtickleHub(id="h2", title="Hub Two"),
    ]
    shared = MindtickleAssetSummary(
        id="a1", title="Shared Guide", last_updated_time=1_700_000_000
    )
    only_two = MindtickleAssetSummary(id="a2", title="Only Two")
    client.list_hub_assets.side_effect = lambda hub_id: (
        [shared] if hub_id == "h1" else [shared, only_two]
    )
    client.get_asset.side_effect = lambda asset_id: MindtickleAssetDetails(
        id=asset_id,
        name=f"Asset {asset_id}",
        description="<p>Hello <b>world</b></p>",
        sharing_type="INTERNAL",
        sharable_link=f"https://deeplinks.mindtickle.com/{asset_id}",
        latest_version=2,
        last_updated_time=1_700_000_000,
        attributes={
            "attributes_count": 1,
            "values": [
                {"category_name": "Resource Type", "attribute_name": "Guideline"}
            ],
        },
    )
    client.get_asset_media.side_effect = lambda asset_id: MindtickleAssetMedia(
        asset_id=asset_id,
        transcript_url="https://cf.example/transcript",
        asset_type="MEDIA_TYPE_DOCUMENT_PDF",
    )
    client.download.return_value = b"Transcript body"

    documents = _documents(connector.load_from_state())

    assert len(documents) == 2
    by_id = {doc.id: doc for doc in documents}
    shared_doc = by_id["MINDTICKLE_ASSET_a1"]
    assert shared_doc.source == DocumentSource.MINDTICKLE
    assert shared_doc.semantic_identifier == "Asset a1"
    assert shared_doc.metadata["hubs"] == ["Hub One", "Hub Two"]
    assert shared_doc.metadata["attributes"] == ["Resource Type: Guideline"]
    assert shared_doc.metadata["asset_type"] == "MEDIA_TYPE_DOCUMENT_PDF"
    assert shared_doc.sections[0].link == "https://deeplinks.mindtickle.com/a1"
    assert shared_doc.sections[0].text == "Hello world\n\nTranscript body"
    assert shared_doc.doc_updated_at == datetime.fromtimestamp(
        1_700_000_000, tz=timezone.utc
    )
    # Each unique asset is fetched exactly once.
    assert client.get_asset.call_count == 2


def test_poll_source_filters_by_last_updated_time() -> None:
    connector, client = _connector_with_mock_client()
    client.list_hubs.return_value = [MindtickleHub(id="h1", title="Hub")]
    client.list_hub_assets.return_value = [
        MindtickleAssetSummary(id="old", title="Old", last_updated_time=100),
        MindtickleAssetSummary(id="new", title="New", last_updated_time=500),
        MindtickleAssetSummary(id="unknown", title="Unknown"),
    ]
    client.get_asset.side_effect = lambda asset_id: MindtickleAssetDetails(
        id=asset_id, name=asset_id
    )
    client.get_asset_media.side_effect = lambda asset_id: MindtickleAssetMedia(
        asset_id=asset_id, asset_type="MEDIA_TYPE_IMAGE"
    )

    documents = _documents(connector.poll_source(400, 600))

    assert sorted(doc.id for doc in documents) == [
        "MINDTICKLE_ASSET_new",
        "MINDTICKLE_ASSET_unknown",
    ]
    # No transcript and no extractable type: the title is the fallback text.
    assert all(doc.sections[0].text == doc.semantic_identifier for doc in documents)


def test_asset_fetch_failure_is_skipped() -> None:
    connector, client = _connector_with_mock_client()
    client.list_hubs.return_value = [MindtickleHub(id="h1", title="Hub")]
    client.list_hub_assets.return_value = [
        MindtickleAssetSummary(id="bad", title="Bad"),
        MindtickleAssetSummary(id="good", title="Good"),
    ]

    def get_asset(asset_id: str) -> MindtickleAssetDetails:
        if asset_id == "bad":
            raise MindtickleClientError("boom", status_code=500)
        return MindtickleAssetDetails(id=asset_id, name="Good")

    client.get_asset.side_effect = get_asset
    client.get_asset_media.return_value = MindtickleAssetMedia(asset_id="good")

    documents = _documents(connector.load_from_state())
    assert [doc.id for doc in documents] == ["MINDTICKLE_ASSET_good"]


def test_slim_docs_list_every_asset_once() -> None:
    connector, client = _connector_with_mock_client()
    client.list_hubs.return_value = [
        MindtickleHub(id="h1", title="Hub One"),
        MindtickleHub(id="h2", title="Hub Two"),
    ]
    shared = MindtickleAssetSummary(id="a1", title="Shared")
    client.list_hub_assets.return_value = [shared]

    slim_ids = [
        doc.id
        for batch in connector.retrieve_all_slim_docs()
        for doc in batch
        if isinstance(doc, SlimDocument)
    ]
    assert slim_ids == ["MINDTICKLE_ASSET_a1"]
