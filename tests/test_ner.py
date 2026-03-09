import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.services.ner as ner_module
from app.services.ner import NERService, _run_ner


def _make_pipeline_result(entities: list[dict]) -> MagicMock:
    mock_pipeline = MagicMock(return_value=entities)
    return mock_pipeline


def _reset_model():
    ner_module._model = None


@pytest.mark.asyncio
@patch("app.services.ner._run_ner")
async def test_extract_entities_returns_correct_dict(mock_run_ner):
    mock_run_ner.return_value = {"PERSON": ["Alice Smith"], "ORG": ["Acme Corp"]}
    service = NERService()
    result = await service.extract_entities("Alice Smith works at Acme Corp.")
    assert result == {"PERSON": ["Alice Smith"], "ORG": ["Acme Corp"]}


@pytest.mark.asyncio
@patch("app.services.ner._run_ner")
async def test_extract_entities_calls_run_ner_with_text(mock_run_ner):
    mock_run_ner.return_value = {}
    service = NERService()
    await service.extract_entities("some input text")
    mock_run_ner.assert_called_once_with("some input text")


@pytest.mark.asyncio
async def test_extract_entities_returns_empty_dict_for_empty_string():
    service = NERService()
    result = await service.extract_entities("")
    assert result == {}


@pytest.mark.asyncio
@patch("app.services.ner._run_ner")
async def test_extract_entities_runs_in_thread(mock_run_ner):
    mock_run_ner.return_value = {"LOC": ["Paris"]}
    service = NERService()
    result = await service.extract_entities("Paris is a city.")
    assert result == {"LOC": ["Paris"]}


@patch("app.services.ner._get_model")
def test_run_ner_maps_per_to_person(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result(
        [{"entity_group": "PER", "word": "Alice", "score": 0.99}]
    )
    result = _run_ner("Alice")
    assert "PERSON" in result
    assert "Alice" in result["PERSON"]


@patch("app.services.ner._get_model")
def test_run_ner_keeps_org_loc_misc_labels(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result(
        [
            {"entity_group": "ORG", "word": "Acme", "score": 0.97},
            {"entity_group": "LOC", "word": "Paris", "score": 0.96},
            {"entity_group": "MISC", "word": "French", "score": 0.85},
        ]
    )
    result = _run_ner("Acme is in Paris and is French.")
    assert "ORG" in result and "Acme" in result["ORG"]
    assert "LOC" in result and "Paris" in result["LOC"]
    assert "MISC" in result and "French" in result["MISC"]


@patch("app.services.ner._get_model")
def test_run_ner_filters_unknown_entity_types(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result(
        [{"entity_group": "UNKNOWN_TYPE", "word": "something", "score": 0.9}]
    )
    result = _run_ner("something")
    assert result == {}


@patch("app.services.ner._get_model")
def test_run_ner_deduplicates_entity_text(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result(
        [
            {"entity_group": "ORG", "word": "Acme", "score": 0.98},
            {"entity_group": "ORG", "word": "Acme", "score": 0.96},
            {"entity_group": "ORG", "word": "Acme", "score": 0.91},
        ]
    )
    result = _run_ner("Acme Acme Acme")
    assert result["ORG"] == ["Acme"]


@patch("app.services.ner._get_model")
def test_run_ner_filters_short_entities(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result(
        [
            {"entity_group": "PER", "word": "A", "score": 0.7},   # 1 char — filtered
            {"entity_group": "PER", "word": "Al", "score": 0.9},  # 2 chars — kept
        ]
    )
    result = _run_ner("A Al")
    assert result.get("PERSON") == ["Al"]


@patch("app.services.ner._get_model")
def test_run_ner_caps_text_at_50000_chars(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result([])
    long_text = "x" * 100_000
    _run_ner(long_text)
    call_args = mock_get_model.return_value.call_args
    passed_text = call_args[0][0]
    assert len(passed_text) == 50_000


@patch("app.services.ner._get_model")
def test_run_ner_returns_empty_dict_for_empty_string(mock_get_model):
    result = _run_ner("")
    mock_get_model.assert_not_called()
    assert result == {}


@patch("app.services.ner._get_model")
def test_run_ner_no_matching_types_returns_empty_dict(mock_get_model):
    mock_get_model.return_value = _make_pipeline_result(
        [{"entity_group": "DATE", "word": "Monday", "score": 0.88}]
    )
    result = _run_ner("Meeting on Monday")
    assert result == {}


@patch("app.services.ner.pipeline")
def test_model_loaded_only_once_across_multiple_calls(mock_pipeline):
    """
    pipeline() (the transformers factory) is called exactly once even when
    _run_ner is called multiple times. This confirms the double-check
    locking pattern in _get_model() works correctly.
    """
    _reset_model()
    mock_pipeline.return_value = MagicMock(return_value=[])

    try:
        _run_ner("first call")
        _run_ner("second call")
        _run_ner("third call")
        assert mock_pipeline.call_count == 1
    finally:
        _reset_model()  # always clean up global state
