from pathlib import Path
from types import SimpleNamespace

import pytest

from src import extractor


EXPECTED = Path(__file__).resolve().parents[1] / "eval" / "expected" / "01_invoice.json"


def test_retries_once_after_invalid_json(monkeypatch) -> None:
    contents = iter(["not JSON", EXPECTED.read_text(encoding="utf-8")])
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=next(contents), thinking=None))

    monkeypatch.setattr(extractor, "chat", fake_chat)
    result, retries = extractor.extract_with_attempts("Счёт", "fake-model")

    assert result.doc_number == "1245"
    assert retries == 1
    assert len(calls) == 2
    assert all(call["think"] is False for call in calls)


def test_fails_after_two_invalid_responses(monkeypatch) -> None:
    def fake_chat(**kwargs):
        return SimpleNamespace(message=SimpleNamespace(content="", thinking=None))

    monkeypatch.setattr(extractor, "chat", fake_chat)
    with pytest.raises(extractor.InvalidModelResponse, match="invalid JSON twice"):
        extractor.extract_with_attempts("Счёт", "fake-model")
