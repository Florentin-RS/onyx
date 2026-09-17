import base64
import io
import json
import zipfile
from unittest.mock import MagicMock

from onyx.connectors.mindtickle.connector import MindtickleConnector
from onyx.connectors.mindtickle.models import (
    MindtickleLearningObject,
    MindtickleModule,
    MindtickleSeries,
)
from onyx.connectors.mindtickle.training import (
    learning_object_to_text,
    rise_runtime_data_to_text,
    scorm_zip_to_text,
)
from onyx.connectors.models import Document, SlimDocument


def _rise_runtime_js(course: dict) -> str:
    encoded = base64.b64encode(json.dumps({"course": course}).encode()).decode()
    return f'__jsonp("runtime-data.js","{encoded}");'


_COURSE = {
    "title": "BDR Battle Cards",
    "description": "<p>How to handle <b>competitors</b> on calls.</p>",
    "lessons": [
        {"type": "section", "title": "Direct Competitors", "items": []},
        {
            "type": "blocks",
            "title": "OpenReel",
            "items": [
                {
                    "type": "text",
                    "settings": {"paddingTop": 3},
                    "items": [
                        {
                            "heading": "Meet your competitor",
                            "paragraph": "<p>A remote recording platform.</p>",
                        }
                    ],
                },
                {
                    "type": "html",
                    "items": [
                        {
                            "srcdoc": "<html><head><style>.x{color:red}</style></head>"
                            "<body><h2>Where they fall short</h2><p>No CFR files.</p>"
                            "<script>var a=1;</script></body></html>"
                        }
                    ],
                },
            ],
        },
    ],
}


def test_rise_runtime_data_to_text_extracts_prose_and_drops_markup() -> None:
    text = rise_runtime_data_to_text(_rise_runtime_js(_COURSE))

    assert text.startswith("BDR Battle Cards\nHow to handle competitors on calls.")
    assert "## Direct Competitors" in text
    assert "## OpenReel" in text
    assert "Meet your competitor" in text
    assert "A remote recording platform." in text
    assert "Where they fall short" in text
    assert "No CFR files." in text
    assert "color:red" not in text
    assert "var a=1" not in text


def test_rise_runtime_data_to_text_without_jsonp_is_empty() -> None:
    assert rise_runtime_data_to_text("console.log('nothing here')") == ""


def _zip(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_scorm_zip_prefers_rise_runtime_data() -> None:
    package = _zip(
        {
            "imsmanifest.xml": "<manifest><title>Manifest title</title></manifest>",
            "scormcontent/index.html": "<html><body>loader shell</body></html>",
            "scormcontent/runtime-data.js": _rise_runtime_js(_COURSE),
        }
    )
    text = scorm_zip_to_text(package)
    assert "Meet your competitor" in text
    assert "loader shell" not in text


def test_scorm_zip_falls_back_to_html_pages() -> None:
    package = _zip(
        {
            "imsmanifest.xml": "<manifest><title>Storyline course</title></manifest>",
            "scormdriver/indexAPI.html": "<html><body>driver</body></html>",
            "story.html": "<html><body><h1>Slide one</h1><p>Body text.</p></body></html>",
        }
    )
    text = scorm_zip_to_text(package)
    assert text.startswith("Storyline course")
    assert "Slide one" in text
    assert "Body text." in text
    assert "driver" not in text


def test_learning_object_quiz_text() -> None:
    lo = MindtickleLearningObject.model_validate(
        {
            "id": "1",
            "type": "Multiple choice question",
            "question": "<p>Which file type avoids sync fixing?</p>",
            "options": [
                {"option": "CFR", "isCorrect": True, "order": 0},
                {"option": "VFR", "isCorrect": False, "order": 1},
            ],
        }
    )
    text = learning_object_to_text(lo, lambda _url, _limit: None)
    assert text == "Which file type avoids sync fixing?\n- CFR\n- VFR"


def test_learning_object_match_labels_text() -> None:
    lo = MindtickleLearningObject.model_validate(
        {
            "id": "2",
            "type": "Match the labels",
            "question": "Match each label",
            "options": [{"question": "Offline-first", "answer": "Works offline"}],
        }
    )
    text = learning_object_to_text(lo, lambda _url, _limit: None)
    assert text == "Match each label\n- Offline-first: Works offline"


def test_learning_object_zip_media_is_downloaded_and_decoded() -> None:
    package = _zip({"scormcontent/runtime-data.js": _rise_runtime_js(_COURSE)})
    lo = MindtickleLearningObject.model_validate(
        {
            "id": "3",
            "type": "Content",
            "media": {
                "id": "m1",
                "type": "Embeded content",
                "url": "https://cf.example/course.zip?Policy=abc",
            },
        }
    )
    downloads: list[str] = []

    def download(url: str, _limit: int) -> bytes:
        downloads.append(url)
        return package

    text = learning_object_to_text(lo, download)
    assert downloads == ["https://cf.example/course.zip?Policy=abc"]
    assert "A remote recording platform." in text


def test_learning_object_iframe_embed_yields_title_and_host() -> None:
    lo = MindtickleLearningObject.model_validate(
        {
            "id": "4",
            "type": "Content",
            "media": {
                "id": "m2",
                "type": "Embeded content",
                "title": "Virtual backgrounds",
                "htmlsrc": '<iframe src="https://www.loom.com/embed/abc"></iframe>',
            },
        }
    )
    text = learning_object_to_text(lo, lambda _url, _limit: None)
    assert text == "Virtual backgrounds (embedded content from www.loom.com)"


def test_learning_object_video_transcript() -> None:
    lo = MindtickleLearningObject.model_validate(
        {
            "id": "5",
            "type": "Content",
            "media": {
                "id": "m3",
                "type": "Video",
                "title": "Zoom positioning",
                "url": "https://cf.example/v.mp4",
                "transcription_url": "https://cf.example/v.json",
            },
        }
    )
    transcript = json.dumps(
        {"results": {"words": [{"word": "All"}, {"word": "right"}, {"word": "team"}]}}
    ).encode()
    text = learning_object_to_text(lo, lambda _url, _limit: transcript)
    assert text == "Zoom positioning\nAll right team"


def _connector_with_mock_client() -> tuple[MindtickleConnector, MagicMock]:
    connector = MindtickleConnector(index_asset_hub=False)
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


def test_modules_are_deduped_across_series_and_unsupported_types_keep_description() -> (
    None
):
    connector, client = _connector_with_mock_client()
    client.list_series.return_value = [
        MindtickleSeries(id="s1", name="AE Onboarding"),
        MindtickleSeries(id="s2", name="Sales Skills Suite"),
    ]
    shared = MindtickleModule(id="m1", name="Value Pillars", module_type="UPDATE")
    checklist = MindtickleModule(
        id="m2",
        name="Chili Piper checklist",
        module_type="CHECKLIST",
        description="<p>Set up Chili Piper.</p>",
    )
    client.list_series_modules.side_effect = lambda series_id: (
        [shared, checklist] if series_id == "s1" else [shared]
    )
    client.get_module_details.side_effect = lambda _series_id, module_id: (
        MindtickleModule(
            id=module_id,
            name="Value Pillars" if module_id == "m1" else "Chili Piper checklist",
            module_type="UPDATE" if module_id == "m1" else "CHECKLIST",
            url=f"https://acme.mindtickle.com/login/jitredirectsso?moduleId={module_id}",
            version=2,
        )
    )
    client.get_module_learning_objects.return_value = [
        MindtickleLearningObject.model_validate(
            {
                "id": "lo1",
                "type": "Text Answer",
                "question": "Name a pillar",
                "exactAnswer": "Reliability",
            }
        )
    ]

    documents = [
        doc
        for batch in connector.load_from_state()
        for doc in batch
        if isinstance(doc, Document)
    ]

    by_id = {doc.id: doc for doc in documents}
    assert set(by_id) == {"MINDTICKLE_MODULE_m1", "MINDTICKLE_MODULE_m2"}
    shared_doc = by_id["MINDTICKLE_MODULE_m1"]
    assert shared_doc.metadata["series"] == ["AE Onboarding", "Sales Skills Suite"]
    assert shared_doc.metadata["kind"] == "training_module"
    assert shared_doc.metadata["module_type"] == "UPDATE"
    assert (
        shared_doc.sections[0].link is not None
        and "moduleId=m1" in shared_doc.sections[0].link
    )
    assert shared_doc.sections[0].text == "Name a pillar\nAnswer: Reliability"
    assert shared_doc.doc_updated_at is None
    # Learning objects are only requested for supported module types.
    client.get_module_learning_objects.assert_called_once_with("m1")
    assert by_id["MINDTICKLE_MODULE_m2"].sections[0].text == "Set up Chili Piper."


def test_unpublished_module_falls_back_to_description() -> None:
    from onyx.connectors.mindtickle.client import MindtickleClientError

    connector, client = _connector_with_mock_client()
    client.list_series.return_value = [MindtickleSeries(id="s1", name="Series")]
    client.list_series_modules.return_value = [
        MindtickleModule(
            id="m1", name="Draft", module_type="UPDATE", description="Soon"
        )
    ]
    client.get_module_details.side_effect = MindtickleClientError(
        "boom", status_code=500
    )
    client.get_module_learning_objects.side_effect = MindtickleClientError(
        "Module state should be published", status_code=400
    )

    documents = [
        doc
        for batch in connector.load_from_state()
        for doc in batch
        if isinstance(doc, Document)
    ]
    assert len(documents) == 1
    assert documents[0].sections[0].text == "Soon"
    assert documents[0].sections[0].link is None


def test_slim_docs_include_modules() -> None:
    connector, client = _connector_with_mock_client()
    client.list_series.return_value = [MindtickleSeries(id="s1", name="Series")]
    client.list_series_modules.return_value = [
        MindtickleModule(id="m1", name="A", module_type="UPDATE")
    ]
    ids = [
        doc.id
        for batch in connector.retrieve_all_slim_docs()
        for doc in batch
        if isinstance(doc, SlimDocument)
    ]
    assert ids == ["MINDTICKLE_MODULE_m1"]
    client.list_hubs.assert_not_called()
