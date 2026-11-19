from pathlib import Path

from src.models import Document


EXPECTED_DIR = Path(__file__).resolve().parents[1] / "eval" / "expected"


def test_expected_json_matches_schema() -> None:
    paths = sorted(EXPECTED_DIR.glob("*.json"))
    assert paths, "No expected JSON files found"

    for path in paths:
        try:
            Document.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise AssertionError(f"Invalid expected JSON: {path.name}") from exc
