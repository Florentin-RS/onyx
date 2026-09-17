from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# ----------------------------------------------------------------- Asset Hub


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


# ----------------------------------------------------------- Training content


class MindtickleSeries(BaseModel):
    """One series as listed by `GET /api/v2/series/list`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    description: str | None = None
    module_type: str | None = Field(
        default=None, validation_alias=AliasChoices("moduleType", "module_type")
    )


class MindtickleModule(BaseModel):
    """One module from `GET /api/v2/series/{id}/list` or the module details call.

    Only the details call returns `url`, the just-in-time SSO link into the module.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    module_type: str = Field(validation_alias=AliasChoices("moduleType", "module_type"))
    description: str | None = None
    thumbnail_url: str | None = Field(
        default=None, validation_alias=AliasChoices("thumbnailUrl", "thumbnail_url")
    )
    url: str | None = None
    version: int | None = None


class MindtickleMedia(BaseModel):
    """Media attached to a learning object. `url` is a signed download; it is empty
    for iframe embeds, which carry the iframe markup in `htmlsrc` instead."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    type: str | None = None
    title: str | None = None
    url: str | None = None
    transcription_url: str | None = None
    htmlsrc: str | None = None


class MindtickleLearningObjectOption(BaseModel):
    """One answer option. Which keys are set depends on the question type."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    option: str | None = None
    text: str | None = None
    question: str | None = None
    answer: str | None = None
    is_correct: bool | None = Field(
        default=None, validation_alias=AliasChoices("isCorrect", "is_correct")
    )
    is_true: bool | None = Field(
        default=None, validation_alias=AliasChoices("isTrue", "is_true")
    )
    order: int | None = None


class MindtickleLearningObject(BaseModel):
    """One entry of `GET /api/v2/entity/{module_id}/learning_objects`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    type: str
    media: MindtickleMedia | None = None
    question: str | None = None
    options: list[MindtickleLearningObjectOption] = Field(default_factory=list)
    answer: str | None = None
    exact_answer: str | None = Field(
        default=None, validation_alias=AliasChoices("exactAnswer", "exact_answer")
    )
    supporting_media_url: str | None = None
