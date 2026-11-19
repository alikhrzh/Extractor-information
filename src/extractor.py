"""Extract a document with Ollama and validate its JSON response."""

import argparse
import os
import sys
from pathlib import Path

from ollama import RequestError, ResponseError, chat
from pydantic import ValidationError

from src.models import Document


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class InvalidModelResponse(ValueError):
    """Both model responses failed JSON schema validation."""


def extract_with_attempts(
    text: str, model: str, prompt_version: str = "v2", think: bool = False
) -> tuple[Document, int]:
    messages = [
        {"role": "system", "content": (PROMPTS_DIR / f"system_{prompt_version}.txt").read_text(encoding="utf-8")},
        {"role": "user", "content": text},
    ]
    schema = Document.model_json_schema()
    first_error = None
    for attempt in range(2):
        response = chat(
            model=model,
            messages=messages,
            format=schema,
            think=think,
            options={"temperature": 0},
        )
        if think and response.message.thinking:
            print(response.message.thinking, file=sys.stderr)
        try:
            return Document.model_validate_json(response.message.content), attempt
        except ValidationError as exc:
            if attempt == 0:
                first_error = exc
                print("Invalid model JSON; retrying once.", file=sys.stderr)
            else:
                raise InvalidModelResponse(
                    f"Model returned invalid JSON twice. First: {first_error}; Second: {exc}"
                ) from exc
    raise RuntimeError("Unreachable")


def extract(text: str, model: str, prompt_version: str = "v2", think: bool = False) -> Document:
    document, _ = extract_with_attempts(text, model, prompt_version, think)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from a text document")
    parser.add_argument("file", type=Path, help="UTF-8 text document")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))
    parser.add_argument("--prompt", choices=("v1", "v2"), default="v2")
    parser.add_argument("--think", action="store_true", help="Show model reasoning")
    args = parser.parse_args(argv)

    try:
        text = args.file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"Ошибка чтения файла: {exc}", file=sys.stderr)
        return 1

    try:
        document = extract(text, args.model, args.prompt, args.think)
    except InvalidModelResponse:
        print("Ошибка: модель дважды вернула невалидный JSON.", file=sys.stderr)
        return 1
    except (OSError, RequestError, ResponseError) as exc:
        print(f"Ошибка вызова Ollama: {exc}", file=sys.stderr)
        return 1

    print(document.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
