"""Run the extractor on all documents and compare its output with the expected JSON."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from src.extractor import extract
from src.models import Document


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "eval" / "documents"
EXPECTED_DIR = ROOT / "eval" / "expected"
FIELDS = list(Document.model_fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction against expected JSON")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--prompt", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()

    documents = sorted(DOCUMENTS_DIR.glob("*.txt"))
    if not documents:
        raise SystemExit("No text documents found")

    correct = {field: 0 for field in FIELDS}
    hallucinations = 0
    omissions = 0
    valid_count = 0
    errors = 0

    for path in documents:
        expected_path = EXPECTED_DIR / f"{path.stem}.json"
        if not expected_path.exists():
            raise SystemExit(f"Missing expected JSON: {expected_path.name}")

        expected = Document.model_validate_json(expected_path.read_text(encoding="utf-8")).model_dump()
        print(f"{path.name}:", flush=True)

        try:
            actual = extract(path.read_text(encoding="utf-8"), args.model, args.prompt).model_dump()
        except (ValidationError, ValueError, OSError) as exc:
            errors += 1
            print(f"  invalid response: {exc}", flush=True)
            continue

        valid_count += 1
        for field in FIELDS:
            if actual[field] == expected[field]:
                correct[field] += 1
            else:
                print(f"  {field}: got {json.dumps(actual[field], ensure_ascii=False)}; "
                      f"expected {json.dumps(expected[field], ensure_ascii=False)}", flush=True)

            # These counts apply to nullable top-level fields only.
            if expected[field] is None and actual[field] is not None:
                hallucinations += 1
            elif expected[field] is not None and actual[field] is None:
                omissions += 1

    print(f"\nValid responses: {valid_count}/{len(documents)}; invalid responses: {errors}")
    print("Field accuracy (exact match, out of all documents):")
    for field in FIELDS:
        print(f"  {field}: {correct[field]}/{len(documents)}")
    print(f"Hallucinations (null expected, value returned): {hallucinations}")
    print(f"Omissions (value expected, null returned): {omissions}")


if __name__ == "__main__":
    main()
