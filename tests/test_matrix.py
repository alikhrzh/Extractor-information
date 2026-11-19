import json
from pathlib import Path
from types import SimpleNamespace

from eval import run_matrix
from src.models import Document
from src.validators import apply_validators


ROOT = Path(__file__).resolve().parents[1]


def test_layer_corrects_bin_and_invented_translation() -> None:
    text = (ROOT / "eval/documents/01_invoice.txt").read_text(encoding="utf-8")
    data = json.loads((ROOT / "eval/expected/01_invoice.json").read_text(encoding="utf-8"))
    data["supplier_bin"] = None
    data["field_confidence"]["supplier_bin"] = 0.0
    data["line_items"][0]["name"] += " / Paper A4"

    corrected, changes = apply_validators(Document.model_validate(data), text)

    assert corrected.supplier_bin == "170340021554"
    assert corrected.line_items[0].name == "Бумага А4 80г/м2"
    assert changes


def test_collection_reuses_saved_answer(monkeypatch, tmp_path) -> None:
    prompts = tmp_path / "prompts"
    documents = tmp_path / "eval/documents"
    expected = tmp_path / "eval/expected"
    for directory in (prompts, documents, expected):
        directory.mkdir(parents=True)
    (prompts / "system_v1.txt").write_text("Extract", encoding="utf-8")
    path = documents / "01_invoice.txt"
    path.write_text("Счёт", encoding="utf-8")
    gold = (ROOT / "eval/expected/01_invoice.json").read_text(encoding="utf-8")
    (expected / "01_invoice.json").write_text(gold, encoding="utf-8")

    monkeypatch.setattr(run_matrix, "ROOT", tmp_path)
    monkeypatch.setattr(run_matrix, "EXPECTED", expected)
    monkeypatch.setattr(run_matrix, "RAW", tmp_path / "eval/raw")
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=gold), done_reason="stop")

    monkeypatch.setattr(run_matrix, "chat", fake_chat)
    first = run_matrix.load_or_collect(path, "v1", "fake-model", collect=True)
    second = run_matrix.load_or_collect(path, "v1", "fake-model", collect=False)

    assert first == second == gold
    assert len(calls) == 1
    assert calls[0]["think"] is False
    assert calls[0]["options"] == {"temperature": 0, "num_ctx": run_matrix.NUM_CTX}
