"""Run the extractor on all documents and compare its output with the expected JSON."""

import argparse
import json
from pathlib import Path

from src.extractor import InvalidModelResponse, extract_with_attempts
from src.models import Document


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "eval" / "documents"
EXPECTED_DIR = ROOT / "eval" / "expected"
FIELDS = list(Document.model_fields)


def evaluate(model: str, prompt: str, think: bool) -> dict:
    documents = sorted(DOCUMENTS_DIR.glob("*.txt"))
    if not documents:
        raise SystemExit("No text documents found")

    print(f"\nMode: think={think}, prompt={prompt}", flush=True)
    correct = {field: 0 for field in FIELDS}
    hallucinations = 0
    omissions = 0
    valid_count = 0
    errors = 0
    retries = 0

    for path in documents:
        expected_path = EXPECTED_DIR / f"{path.stem}.json"
        if not expected_path.exists():
            raise SystemExit(f"Missing expected JSON: {expected_path.name}")

        expected = Document.model_validate_json(expected_path.read_text(encoding="utf-8")).model_dump()
        print(f"{path.name}:", flush=True)

        try:
            result, attempts = extract_with_attempts(
                path.read_text(encoding="utf-8"), model, prompt, think
            )
            actual = result.model_dump()
            retries += attempts
        except InvalidModelResponse as exc:
            retries += 1
            errors += 1
            print(f"  invalid response: {exc}", flush=True)
            continue
        except (ValueError, OSError) as exc:
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
    print(f"Retries attempted: {retries}; final failures: {errors}")
    return {
        "correct": correct,
        "hallucinations": hallucinations,
        "omissions": omissions,
        "valid": valid_count,
        "errors": errors,
        "retries": retries,
        "total": len(documents),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction against expected JSON")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--prompt", choices=("v1", "v2"), default="v2")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.prompt, args.think)


if __name__ == "__main__":
    main()
