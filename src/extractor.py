"""Extract a document with Ollama and validate its JSON response."""

import argparse
import os
import sys
from pathlib import Path

from ollama import chat

from src.models import Document


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def extract(text: str, model: str, prompt_version: str = "v2") -> Document:
    messages = [
        {"role": "system", "content": (PROMPTS_DIR / f"system_{prompt_version}.txt").read_text(encoding="utf-8")},
        {"role": "user", "content": text},
    ]
    schema = Document.model_json_schema()
    response = chat(
        model=model,
        messages=messages,
        format=schema,
        think=True,
        options={"temperature": 0},
    )
    if response.message.thinking:
        print(response.message.thinking, file=sys.stderr)
    content = response.message.content
    if not content or not content.strip():
        print("No final answer; retrying without thinking.", file=sys.stderr)
        response = chat(
            model=model,
            messages=messages,
            format=schema,
            think=False,
            options={"temperature": 0},
        )
        content = response.message.content
    if not content or not content.strip():
        raise ValueError("Ollama returned an empty answer instead of JSON after retry")
    return Document.model_validate_json(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured data from a text document")
    parser.add_argument("file", type=Path, help="UTF-8 text document")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))
    parser.add_argument("--prompt", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8")
    document = extract(text, args.model, args.prompt)
    print(document.model_dump_json(indent=2))
