from io import BytesIO
from typing import Any

from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.constants import DocumentSource
from onyx.connectors.cross_connector_utils.miscellaneous_utils import (
    datetime_from_utc_timestamp,
)
from onyx.connectors.exceptions import (
    ConnectorValidationError,
    CredentialInvalidError,
    UnexpectedValidationError,
)
from onyx.connectors.interfaces import (
    GenerateDocumentsOutput,
    GenerateSlimDocumentOutput,
    LoadConnector,
    PollConnector,
    SecondsSinceUnixEpoch,
    SlimConnector,
)
from onyx.connectors.mindtickle.client import (
    MindtickleAuthenticationError,
    MindtickleClient,
    MindtickleClientError,
)
from onyx.connectors.mindtickle.models import (
    MindtickleAssetDetails,
    MindtickleAssetMedia,
    MindtickleAssetSummary,
    MindtickleHub,
)
from onyx.connectors.mindtickle.utils import (
    asset_type_to_extension,
    filter_hubs,
    html_to_text,
    transcript_to_text,
)
from onyx.connectors.models import (
    ConnectorMissingCredentialError,
    Document,
    HierarchyNode,
    SlimDocument,
    TextSection,
)
from onyx.file_processing.extract_file_text import extract_file_text
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.utils.logger import setup_logger

logger = setup_logger()

_DOC_ID_PREFIX = "MINDTICKLE_ASSET_"
_SLIM_BATCH_SIZE = 1000
_MAX_TRANSCRIPT_BYTES = 20 * 1024 * 1024
_MAX_RAW_CONTENT_BYTES = 50 * 1024 * 1024


class _AssetPlacement:
    """One asset together with every hub it is published in."""

    def __init__(self, summary: MindtickleAssetSummary, hub_title: str) -> None:
        self.summary = summary
        self.hub_titles = [hub_title]


class MindtickleConnector(LoadConnector, PollConnector, SlimConnector):
    """Indexes published assets from Mindtickle Asset Hubs.

    One document per asset. An asset published in several hubs is indexed once,
    with every hub name in its metadata.
    """

    def __init__(
        self,
        hub_names: list[str] | None = None,
        excluded_hub_names: list[str] | None = None,
        batch_size: int = INDEX_BATCH_SIZE,
    ) -> None:
        self.hub_names = [name for name in (hub_names or []) if name.strip()]
        self.excluded_hub_names = [
            name for name in (excluded_hub_names or []) if name.strip()
        ]
        self.batch_size = batch_size

        self._client: MindtickleClient | None = None
        self._api_key: str | None = None
        self._secret_key: str | None = None
        self._learning_site_url: str | None = None

    # ---------------------------------------------------------- credentials

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self._api_key = credentials.get("mindtickle_api_key")
        self._secret_key = credentials.get("mindtickle_secret_key")
        self._learning_site_url = credentials.get("mindtickle_learning_site_url")
        self._client = None
        return None

    @property
    def client(self) -> MindtickleClient:
        if self._client is None:
            if not self._api_key or not self._secret_key or not self._learning_site_url:
                raise ConnectorMissingCredentialError("Mindtickle")
            self._client = MindtickleClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                learning_site_url=self._learning_site_url,
            )
        return self._client

    # ----------------------------------------------------------------- hubs

    def _hubs_to_process(self) -> list[MindtickleHub]:
        hubs = self.client.list_hubs()
        selected = filter_hubs(hubs, self.hub_names, self.excluded_hub_names)
        if self.hub_names and not selected:
            available = ", ".join(sorted(hub.title for hub in hubs))
            raise ConnectorValidationError(
                f"None of the configured hubs {self.hub_names} exist in Mindtickle. "
                f"Available hubs: {available}"
            )
        return selected

    def _collect_assets(self, hubs: list[MindtickleHub]) -> dict[str, _AssetPlacement]:
        """List every hub and group the assets by id."""
        placements: dict[str, _AssetPlacement] = {}
        for hub in hubs:
            assets = self.client.list_hub_assets(hub.id)
            logger.info("Mindtickle hub '%s' lists %d assets", hub.title, len(assets))
            for asset in assets:
                placement = placements.get(asset.id)
                if placement is None:
                    placements[asset.id] = _AssetPlacement(asset, hub.title)
                    continue
                placement.hub_titles.append(hub.title)
                if (asset.last_updated_time or 0) > (
                    placement.summary.last_updated_time or 0
                ):
                    placement.summary = asset
        return placements

    # ------------------------------------------------------------ documents

    @staticmethod
    def _in_window(
        summary: MindtickleAssetSummary,
        start: SecondsSinceUnixEpoch | None,
        end: SecondsSinceUnixEpoch | None,
    ) -> bool:
        updated = summary.last_updated_time
        if updated is None:
            # Unknown update time: never silently drop the asset.
            return True
        if start is not None and updated < start:
            return False
        if end is not None and updated > end:
            return False
        return True

    def _fetch_transcript_text(
        self, details: MindtickleAssetDetails, media: MindtickleAssetMedia
    ) -> str:
        """Prefer Mindtickle's pre-extracted transcript; fall back to the raw file."""
        if media.transcript_url:
            raw = self.client.download(media.transcript_url, _MAX_TRANSCRIPT_BYTES)
            if raw is None:
                logger.warning(
                    "Mindtickle transcript for asset %s exceeds size limit", details.id
                )
            else:
                text = transcript_to_text(raw)
                if text:
                    return text

        extension = asset_type_to_extension(media.asset_type)
        if not extension or not media.raw_content_url:
            return ""
        raw = self.client.download(media.raw_content_url, _MAX_RAW_CONTENT_BYTES)
        if raw is None:
            logger.warning(
                "Mindtickle raw content for asset %s exceeds size limit", details.id
            )
            return ""
        file_name = (
            details.name
            if details.name.lower().endswith(extension)
            else (f"{details.name}{extension}")
        )
        return extract_file_text(BytesIO(raw), file_name, break_on_unprocessable=False)

    @staticmethod
    def _build_metadata(
        details: MindtickleAssetDetails,
        media: MindtickleAssetMedia,
        hub_titles: list[str],
    ) -> dict[str, str | list[str]]:
        metadata: dict[str, str | list[str]] = {
            "hubs": sorted(set(hub_titles)),
        }
        if details.sharing_type:
            metadata["sharing_type"] = details.sharing_type
        if media.asset_type:
            metadata["asset_type"] = media.asset_type
        if details.latest_version is not None:
            metadata["version"] = str(details.latest_version)
        attributes = sorted(
            {
                f"{value.category_name}: {value.attribute_name}"
                for value in details.attributes.values
                if value.category_name and value.attribute_name
            }
        )
        if attributes:
            metadata["attributes"] = attributes
        return metadata

    def _build_document(self, placement: _AssetPlacement) -> Document | None:
        asset_id = placement.summary.id
        try:
            details = self.client.get_asset(asset_id)
            media = self.client.get_asset_media(asset_id)
        except MindtickleClientError as e:
            logger.warning("Skipping Mindtickle asset %s: %s", asset_id, e)
            return None

        try:
            body = self._fetch_transcript_text(details, media)
        except MindtickleClientError as e:
            logger.warning(
                "Mindtickle asset %s content download failed, indexing metadata only: %s",
                asset_id,
                e,
            )
            body = ""
        except Exception:
            logger.exception(
                "Mindtickle asset %s text extraction failed, indexing metadata only",
                asset_id,
            )
            body = ""

        title = details.name.strip() or placement.summary.title.strip() or asset_id
        description = html_to_text(details.description or placement.summary.description)
        text_parts = [part for part in (description, body) if part]
        text = "\n\n".join(text_parts) if text_parts else title

        updated = details.last_updated_time or placement.summary.last_updated_time
        return Document(
            id=f"{_DOC_ID_PREFIX}{asset_id}",
            sections=[TextSection(link=details.sharable_link, text=text)],
            source=DocumentSource.MINDTICKLE,
            semantic_identifier=title,
            metadata=self._build_metadata(details, media, placement.hub_titles),
            doc_updated_at=(
                datetime_from_utc_timestamp(updated) if updated is not None else None
            ),
        )

    def _generate_documents(
        self,
        start: SecondsSinceUnixEpoch | None,
        end: SecondsSinceUnixEpoch | None,
    ) -> GenerateDocumentsOutput:
        placements = self._collect_assets(self._hubs_to_process())
        candidates = [
            placement
            for placement in placements.values()
            if self._in_window(placement.summary, start, end)
        ]
        logger.info(
            "Mindtickle: %d unique assets, %d within the requested window",
            len(placements),
            len(candidates),
        )

        batch: list[Document | HierarchyNode] = []
        for placement in candidates:
            document = self._build_document(placement)
            if document is None:
                continue
            batch.append(document)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ----------------------------------------------------------- interfaces

    def load_from_state(self) -> GenerateDocumentsOutput:
        return self._generate_documents(None, None)

    def poll_source(
        self, start: SecondsSinceUnixEpoch, end: SecondsSinceUnixEpoch
    ) -> GenerateDocumentsOutput:
        return self._generate_documents(start, end)

    def retrieve_all_slim_docs(
        self,
        start: SecondsSinceUnixEpoch | None = None,  # noqa: ARG002
        end: SecondsSinceUnixEpoch | None = None,  # noqa: ARG002
        callback: IndexingHeartbeatInterface | None = None,  # noqa: ARG002
    ) -> GenerateSlimDocumentOutput:
        placements = self._collect_assets(self._hubs_to_process())
        batch: list[SlimDocument | HierarchyNode] = []
        for asset_id in placements:
            batch.append(SlimDocument(id=f"{_DOC_ID_PREFIX}{asset_id}"))
            if len(batch) >= _SLIM_BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    def validate_connector_settings(self) -> None:
        if not self._api_key or not self._secret_key or not self._learning_site_url:
            raise ConnectorMissingCredentialError("Mindtickle")
        try:
            self._hubs_to_process()
        except MindtickleAuthenticationError as e:
            raise CredentialInvalidError(str(e)) from e
        except ConnectorValidationError:
            raise
        except MindtickleClientError as e:
            if e.status_code == 403:
                raise ConnectorValidationError(
                    f"The Mindtickle credentials cannot read Asset Hubs: {e}"
                ) from e
            raise UnexpectedValidationError(
                f"Unexpected Mindtickle error during validation: {e}"
            ) from e


if __name__ == "__main__":
    import os

    connector = MindtickleConnector(
        hub_names=[
            name
            for name in os.environ.get("MINDTICKLE_HUB_NAMES", "").split(",")
            if name
        ]
    )
    connector.load_credentials(
        {
            "mindtickle_api_key": os.environ["MINDTICKLE_API_KEY"],
            "mindtickle_secret_key": os.environ["MINDTICKLE_SECRET_KEY"],
            "mindtickle_learning_site_url": os.environ["MINDTICKLE_LEARNING_SITE_URL"],
        }
    )
    for doc_batch in connector.load_from_state():
        for doc in doc_batch:
            print(doc.to_short_descriptor() if isinstance(doc, Document) else doc)
