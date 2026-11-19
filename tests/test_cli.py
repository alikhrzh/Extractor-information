import json
from pathlib import Path
from types import SimpleNamespace

from src import extractor


ROOT = Path(__file__).resolve().parents[1]


def test_cli_prints_only_valid_json(monkeypatch, capsys) -> None:
    expected = (ROOT / "eval/expected/01_invoice.json").read_text(encoding="utf-8")

    def fake_chat(**kwargs):
        return SimpleNamespace(message=SimpleNamespace(content=expected, thinking=None))

    monkeypatch.setattr(extractor, "chat", fake_chat)
    path = ROOT / "eval/documents/01_invoice.txt"
    code = extractor.main([str(path)])

    output = capsys.readouterr()
    assert code == 0
    assert json.loads(output.out)["doc_number"] == "1245"
    assert output.err == ""


def test_cli_reports_invalid_model_answer_without_traceback(monkeypatch, capsys) -> None:
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="", thinking=None))

    monkeypatch.setattr(extractor, "chat", fake_chat)
    path = ROOT / "eval/documents/01_invoice.txt"
    code = extractor.main([str(path)])

    output = capsys.readouterr()
    assert code == 1
    assert len(calls) == 2
    assert output.out == ""
    assert "дважды вернула невалидный JSON" in output.err
    assert "Traceback" not in output.err
