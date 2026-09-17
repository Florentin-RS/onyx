from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class MindtickleHub(BaseModel):
    """One Asset Hub as returned by `POST /api/assethub/v1/hubs`."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    description: str | None = None
    last_updated_at: int | None = None


class MindtickleAssetSummary(BaseModel):
    """One asset as listed by `POST /api/assethub/v1/hub/{hub_id}/assets`.

    The docs name the title field `name`; the live API returns `title`.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    title: str = Field(validation_alias=AliasChoices("title", "name"))
    description: str | None = None
    sharing_type: str | None = None
    last_updated_time: int | None = None
    expiry_time: int | None = None


class MindtickleAssetAttribute(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category_name: str | None = None
    attribute_id: str | None = None
    attribute_name: str | None = None


class MindtickleAssetAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attributes_count: int = 0
    values: list[MindtickleAssetAttribute] = Field(default_factory=list)


class MindtickleAssetDetails(BaseModel):
    """Response of `POST /api/assethub/v1/asset/{asset_id}`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str = Field(validation_alias=AliasChoices("name", "title"))
    description: str | None = None
    sharing_type: str | None = None
    sharable_link: str | None = None
    latest_version: int | None = None
    last_updated_time: int | None = None
    expiry_time: int | None = None
    attributes: MindtickleAssetAttributes = Field(
        default_factory=MindtickleAssetAttributes
    )


class MindtickleAssetMedia(BaseModel):
    """Response of `POST /api/assethub/v1/assetmedia/{asset_id}`.

    All URLs are signed and time-bound.
    """

    model_config = ConfigDict(extra="ignore")

    asset_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    raw_content_url: str | None = None
    transcript_url: str | None = None
    asset_type: str | None = None
