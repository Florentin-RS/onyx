import time
from typing import Any

import requests
from requests.exceptions import RequestException

from onyx.configs.app_configs import REQUEST_TIMEOUT_SECONDS
from onyx.connectors.cross_connector_utils.rate_limit_wrapper import (
    rate_limit_builder,
)
from onyx.connectors.mindtickle.models import (
    MindtickleAssetDetails,
    MindtickleAssetMedia,
    MindtickleAssetSummary,
    MindtickleHub,
    MindtickleLearningObject,
    MindtickleModule,
    MindtickleSeries,
)
from onyx.utils.logger import setup_logger
from onyx.utils.retry_after import parse_retry_after_seconds

logger = setup_logger()

# Mindtickle serves its APIs from two host groups. Training content and user
# management use the standard hosts; Asset Hub content uses the "API3" hosts. A
# bearer token is only valid on the host that issued it. The global host is tried
# first; a learning site hosted in the US answers 404 there.
MINDTICKLE_API_BASE_URLS: tuple[str, ...] = (
    "https://api.mindtickle.com",
    "https://api.prod-us.mindtickle.com",
)
MINDTICKLE_API3_BASE_URLS: tuple[str, ...] = (
    "https://api3.prod.mindtickle.com",
    "https://api3.prod-us.mindtickle.com",
)

_AUTH_PATH = "/services/data/auth_token"
_HUBS_PATH = "/api/assethub/v1/hubs"
_HUB_ASSETS_PATH = "/api/assethub/v1/hub/{hub_id}/assets"
_ASSET_PATH = "/api/assethub/v1/asset/{asset_id}"
_ASSET_MEDIA_PATH = "/api/assethub/v1/assetmedia/{asset_id}"
_SERIES_LIST_PATH = "/api/v2/series/list"
_SERIES_MODULES_PATH = "/api/v2/series/{series_id}/list"
_MODULE_DETAILS_PATH = "/api/v2/jit/series/{series_id}/entity/{module_id}"
_MODULE_LEARNING_OBJECTS_PATH = "/api/v2/entity/{module_id}/learning_objects"

# Server-side maximum page size.
_PAGE_SIZE = 100
# Tokens last 3600 s. Refresh a little early so a long page loop never trips a 401.
_TOKEN_REFRESH_MARGIN_SECONDS = 300
# Platform limit is 5 req/s and 220 req/min. Stay under both.
_MAX_REQUESTS_PER_SECOND = 3
_MAX_RATE_LIMIT_WAITS = 10
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 15.0
_MAX_RATE_LIMIT_WAIT_SECONDS = 120.0
_MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 120
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class MindtickleClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class MindtickleAuthenticationError(MindtickleClientError):
    """The credentials were rejected, or the learning site exists in no region."""


class MindtickleRateLimitError(MindtickleClientError):
    """The API kept returning 429 after every allowed wait."""


def normalize_learning_site_url(value: str) -> str:
    """Return the bare host, e.g. `acme.mindtickle.com`, from any pasted form."""
    host = value.strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0]
    return host.strip().lower()


@rate_limit_builder(max_calls=_MAX_REQUESTS_PER_SECOND, period=1)
def _send(
    session: requests.Session,
    method: str,
    url: str,
    body: dict[str, Any] | None,
    token: str,
    timeout: int,
) -> requests.Response:
    """Every authenticated API call goes through here so one limiter covers both
    host groups (the platform quota is per account, not per host)."""
    return session.request(
        method,
        url,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )


class _ApiHost:
    """Token lifecycle and region detection for one Mindtickle host group."""

    def __init__(
        self,
        base_urls: tuple[str, ...],
        session: requests.Session,
        api_key: str,
        secret_key: str,
        learning_site_url: str,
        timeout: int,
    ) -> None:
        self._base_urls = base_urls
        self._session = session
        self._api_key = api_key
        self._secret_key = secret_key
        self._learning_site_url = learning_site_url
        self._timeout = timeout
        self.base_url: str | None = None
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def token(self, force_refresh: bool = False) -> str:
        if (
            not force_refresh
            and self._token is not None
            and time.monotonic()
            < self._token_expires_at - _TOKEN_REFRESH_MARGIN_SECONDS
        ):
            return self._token
        return self._authenticate()

    def _authenticate(self) -> str:
        """Exchange the credentials for a bearer token.

        When the region is not yet known every base URL is tried. A wrong region
        answers 404 "Learning Site not found!". Once a host works it is kept.
        """
        candidates = (self.base_url,) if self.base_url else self._base_urls
        body = {
            "api_key": self._api_key,
            "secret_key": self._secret_key,
            "ls_url": self._learning_site_url,
        }
        last_error: str | None = None
        for base_url in candidates:
            if base_url is None:
                continue
            try:
                response = self._session.post(
                    base_url + _AUTH_PATH, json=body, timeout=self._timeout
                )
            except RequestException as e:
                last_error = f"{base_url}: {e}"
                logger.warning("Mindtickle auth request failed: %s", last_error)
                continue

            if response.status_code == 200:
                payload = response.json()
                token = payload.get("token")
                if not token:
                    raise MindtickleAuthenticationError(
                        "Mindtickle auth response did not include a token"
                    )
                expires_in = float(payload.get("expires_in") or 3600)
                self.base_url = base_url
                self._token = token
                self._token_expires_at = time.monotonic() + expires_in
                logger.info("Authenticated with Mindtickle at %s", base_url)
                return token

            if response.status_code == 404:
                # Learning site is hosted in another region. Try the next host.
                last_error = f"{base_url}: {response.text.strip()[:200]}"
                continue

            if response.status_code in (401, 403):
                raise MindtickleAuthenticationError(
                    f"Mindtickle rejected the API key or secret key "
                    f"(HTTP {response.status_code})",
                    status_code=response.status_code,
                )

            raise MindtickleClientError(
                f"Mindtickle auth failed with HTTP {response.status_code}: "
                f"{response.text.strip()[:200]}",
                status_code=response.status_code,
            )

        raise MindtickleAuthenticationError(
            f"Learning site '{self._learning_site_url}' was not found in any "
            f"Mindtickle region. Last error: {last_error}",
            status_code=404,
        )

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send one API call and return the parsed JSON object.

        Handles token refresh on 401 and Retry-After on 429.
        """
        token = self.token()
        assert self.base_url is not None
        url = self.base_url + path
        refreshed = False

        for _ in range(_MAX_RATE_LIMIT_WAITS):
            try:
                response = _send(self._session, method, url, body, token, self._timeout)
            except RequestException as e:
                raise MindtickleClientError(f"Request to {path} failed: {e}") from e

            if response.status_code == 429:
                parsed = parse_retry_after_seconds(response.headers.get("Retry-After"))
                wait = (
                    parsed if parsed is not None else _DEFAULT_RATE_LIMIT_WAIT_SECONDS
                )
                wait = min(wait, _MAX_RATE_LIMIT_WAIT_SECONDS)
                logger.notice(
                    "Mindtickle rate limit hit on %s. Waiting %s seconds.", path, wait
                )
                time.sleep(wait)
                continue

            if response.status_code == 401 and not refreshed:
                refreshed = True
                token = self.token(force_refresh=True)
                continue

            if response.status_code >= 400:
                raise MindtickleClientError(
                    f"Mindtickle API error on {path}: HTTP {response.status_code} "
                    f"{response.text.strip()[:300]}",
                    status_code=response.status_code,
                )

            if not response.content or not response.content.strip():
                return {}
            data = response.json()
            if not isinstance(data, dict):
                raise MindtickleClientError(
                    f"Unexpected non-object response from {path}"
                )
            return data

        raise MindtickleRateLimitError(
            f"Mindtickle kept rate limiting {path} after {_MAX_RATE_LIMIT_WAITS} waits",
            status_code=429,
        )


class MindtickleClient:
    """Client for the Mindtickle Asset Hub and Training Content APIs."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        learning_site_url: str,
        api_base_urls: tuple[str, ...] = MINDTICKLE_API_BASE_URLS,
        api3_base_urls: tuple[str, ...] = MINDTICKLE_API3_BASE_URLS,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key or not secret_key or not learning_site_url:
            raise ValueError(
                "Mindtickle API key, secret key and learning site URL are required"
            )
        self._session = requests.Session()
        site = normalize_learning_site_url(learning_site_url)
        self._training_host = _ApiHost(
            api_base_urls, self._session, api_key, secret_key, site, timeout
        )
        self._asset_hub_host = _ApiHost(
            api3_base_urls, self._session, api_key, secret_key, site, timeout
        )

    # ------------------------------------------------------------- Asset Hub

    def _paginate(
        self,
        path: str,
        items_key: str,
        total_key: str,
        order_field: str,
    ) -> list[dict[str, Any]]:
        """Walk a skip/limit list endpoint to the end.

        The items key is absent when a page is empty, so both the total count and
        an empty page end the loop.
        """
        items: list[dict[str, Any]] = []
        skip = 0
        while True:
            body: dict[str, Any] = {
                "skip": skip,
                "limit": _PAGE_SIZE,
                "order_by": {"field": order_field, "order": "ASC"},
            }
            data = self._asset_hub_host.request("POST", path, body)
            page = data.get(items_key) or []
            if not isinstance(page, list):
                raise MindtickleClientError(
                    f"Unexpected '{items_key}' payload from {path}"
                )
            items.extend(item for item in page if isinstance(item, dict))
            skip += len(page)

            total_raw = data.get(total_key)
            total = int(total_raw) if isinstance(total_raw, (int, str)) else None
            if (
                not page
                or len(page) < _PAGE_SIZE
                or (total is not None and skip >= total)
            ):
                return items

    def list_hubs(self) -> list[MindtickleHub]:
        raw_hubs = self._paginate(
            _HUBS_PATH,
            items_key="hub_details",
            total_key="total_hubs_count",
            order_field="title",
        )
        return [MindtickleHub.model_validate(hub) for hub in raw_hubs]

    def list_hub_assets(self, hub_id: str) -> list[MindtickleAssetSummary]:
        raw_assets = self._paginate(
            _HUB_ASSETS_PATH.format(hub_id=hub_id),
            items_key="ref_asset",
            total_key="total_count",
            order_field="updated_at",
        )
        return [MindtickleAssetSummary.model_validate(asset) for asset in raw_assets]

    def get_asset(self, asset_id: str) -> MindtickleAssetDetails:
        data = self._asset_hub_host.request(
            "POST", _ASSET_PATH.format(asset_id=asset_id), {}
        )
        return MindtickleAssetDetails.model_validate(data)

    def get_asset_media(self, asset_id: str) -> MindtickleAssetMedia:
        data = self._asset_hub_host.request(
            "POST", _ASSET_MEDIA_PATH.format(asset_id=asset_id), {}
        )
        return MindtickleAssetMedia.model_validate(data)

    # ------------------------------------------------------ Training content

    @staticmethod
    def _hits(data: dict[str, Any], path: str) -> list[dict[str, Any]]:
        hits = data.get("hits") or []
        if not isinstance(hits, list):
            raise MindtickleClientError(f"Unexpected 'hits' payload from {path}")
        return [hit for hit in hits if isinstance(hit, dict)]

    def list_series(self) -> list[MindtickleSeries]:
        data = self._training_host.request("GET", _SERIES_LIST_PATH)
        return [
            MindtickleSeries.model_validate(hit)
            for hit in self._hits(data, _SERIES_LIST_PATH)
        ]

    def list_series_modules(self, series_id: str) -> list[MindtickleModule]:
        path = _SERIES_MODULES_PATH.format(series_id=series_id)
        data = self._training_host.request("GET", path)
        return [MindtickleModule.model_validate(hit) for hit in self._hits(data, path)]

    def get_module_details(self, series_id: str, module_id: str) -> MindtickleModule:
        path = _MODULE_DETAILS_PATH.format(series_id=series_id, module_id=module_id)
        return MindtickleModule.model_validate(self._training_host.request("GET", path))

    def get_module_learning_objects(
        self, module_id: str
    ) -> list[MindtickleLearningObject]:
        """Learning objects of a published Course, Quick Update or Assessment.

        The docs call the list `learning_contents`; the live API returns
        `learningObjects`. Unsupported module types and drafts answer HTTP 400,
        which the caller treats as "no learning objects".
        """
        path = _MODULE_LEARNING_OBJECTS_PATH.format(module_id=module_id)
        data = self._training_host.request("GET", path)
        raw = data.get("learningObjects")
        if raw is None:
            raw = data.get("learning_contents") or []
        if not isinstance(raw, list):
            raise MindtickleClientError(
                f"Unexpected learning object payload from {path}"
            )
        return [
            MindtickleLearningObject.model_validate(item)
            for item in raw
            if isinstance(item, dict)
        ]

    # -------------------------------------------------------------- downloads

    def download(self, url: str, max_bytes: int) -> bytes | None:
        """Download a signed media URL. Returns None when the body exceeds max_bytes.

        Signed CloudFront URLs carry their own auth, so no bearer token is sent.
        """
        try:
            with requests.get(
                url, stream=True, timeout=_MEDIA_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                if response.status_code >= 400:
                    raise MindtickleClientError(
                        f"Media download failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    return None
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    size += len(chunk)
                    if size > max_bytes:
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
        except RequestException as e:
            raise MindtickleClientError(f"Media download failed: {e}") from e
