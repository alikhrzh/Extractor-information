"""Collect one raw answer per prompt/document, then score four offline variants."""

import argparse
import hashlib
import json
from pathlib import Path

from ollama import chat
from pydantic import ValidationError

from src.models import Document
from src.validators import apply_validators


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "eval" / "documents"
EXPECTED = ROOT / "eval" / "expected"
RAW = ROOT / "eval" / "raw"
NUM_CTX = 8192
FIELDS = list(Document.model_fields)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare() -> list[Path]:
    paths = sorted(DOCUMENTS.glob("*.txt"))
    if not paths:
        raise ValueError("No documents found")
    for path in paths:
        gold = EXPECTED / f"{path.stem}.json"
        if not gold.exists():
            raise ValueError(f"Missing expected JSON: {gold}")
        Document.model_validate_json(gold.read_text(encoding="utf-8"))
    return paths


def load_or_collect(path: Path, version: str, model: str, collect: bool) -> str:
    document_text = path.read_text(encoding="utf-8")
    expected_text = (EXPECTED / f"{path.stem}.json").read_text(encoding="utf-8")
    prompt_text = (ROOT / "prompts" / f"system_{version}.txt").read_text(encoding="utf-8")
    raw_path = RAW / version / f"{path.stem}.json"

    if raw_path.exists():
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        for key, value in {
            "model": model,
            "prompt_sha256": digest(prompt_text),
            "document_sha256": digest(document_text),
            "expected_sha256": digest(expected_text),
            "num_ctx": NUM_CTX,
        }.items():
            if record.get(key) != value:
                raise ValueError(f"{raw_path}: {key} changed since collection")
        return record["content"]

    if not collect:
        raise ValueError(f"Missing raw answer: {raw_path}; run --collect first")

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": document_text},
        ],
        format=Document.model_json_schema(),
        think=False,
        options={"temperature": 0, "num_ctx": NUM_CTX},
    )
    record = {
        "model": model,
        "prompt_sha256": digest(prompt_text),
        "document_sha256": digest(document_text),
        "expected_sha256": digest(expected_text),
        "num_ctx": NUM_CTX,
        "content": response.message.content or "",
        "done_reason": str(getattr(response, "done_reason", None)),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {version}/{path.stem}", flush=True)
    return record["content"]


def new_counts() -> dict:
    return {"correct": {field: 0 for field in FIELDS}, "hallucinations": 0,
            "omissions": 0, "invalid": 0, "corrections": 0, "total": 0}


def score(counts: dict, actual: Document | None, expected: Document) -> None:
    counts["total"] += 1
    if actual is None:
        counts["invalid"] += 1
        return
    got = actual.model_dump()
    gold = expected.model_dump()
    for field in FIELDS:
        if got[field] == gold[field]:
            counts["correct"][field] += 1
        if gold[field] is None and got[field] is not None:
            counts["hallucinations"] += 1
        elif gold[field] is not None and got[field] is None:
            counts["omissions"] += 1


def report(group: str, cells: dict) -> None:
    print(f"\n{group}: A=v1 raw, B=v1 layer, C=v2 raw, D=v2 layer")
    print("field                         A       B       C       D")
    for field in FIELDS:
        scores = [f"{cells[key]['correct'][field]}/{cells[key]['total']}" for key in "ABCD"]
        print(f"{field:28} {'  '.join(f'{value:>5}' for value in scores)}")
    for metric in ("hallucinations", "omissions", "invalid", "corrections"):
        print(f"{metric:28} {'  '.join(f'{cells[key][metric]:>5}' for key in 'ABCD')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1/v2 with the same saved model answers")
    parser.add_argument("--collect", action="store_true", help="Call Ollama for missing raw answers")
    parser.add_argument("--model", default="qwen3.5:4b")
    args = parser.parse_args()

    paths = prepare()
    groups = {group: {key: new_counts() for key in "ABCD"} for group in ("dev", "holdout", "all")}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        expected = Document.model_validate_json((EXPECTED / f"{path.stem}.json").read_text(encoding="utf-8"))
        group = "dev" if int(path.stem.split("_", 1)[0]) <= 3 else "holdout"
        for version, raw_key, layer_key in (("v1", "A", "B"), ("v2", "C", "D")):
            content = load_or_collect(path, version, args.model, args.collect)
            try:
                raw = Document.model_validate_json(content)
            except ValidationError as exc:
                print(f"Invalid {version}/{path.stem}: {exc.errors()[0]['msg']}")
                raw = None
            layered, corrections = apply_validators(raw, text) if raw else (None, [])
            for bucket in (groups[group], groups["all"]):
                score(bucket[raw_key], raw, expected)
                score(bucket[layer_key], layered, expected)
                bucket[layer_key]["corrections"] += len(corrections)
            for correction in corrections:
                print(f"{version}/{path.stem}: {correction}")

    for group in ("dev", "holdout", "all"):
        report(group, groups[group])


if __name__ == "__main__":
    main()

