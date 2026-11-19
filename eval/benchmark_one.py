"""Measure one Ollama extraction call without changing saved matrix answers."""

from pathlib import Path

from ollama import chat

from src.models import Document


def main() -> None:
    text = Path("eval/documents/01_invoice.txt").read_text(encoding="utf-8")
    prompt = Path("prompts/system_v2.txt").read_text(encoding="utf-8")
    response = chat(
        model="qwen3.5:4b",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        format=Document.model_json_schema(),
        think=False,
        options={"temperature": 0, "num_ctx": 8192},
    )
    seconds = response.total_duration / 1_000_000_000
    print(f"Время: {seconds:.1f} с")
    print(f"Токены: вход {response.prompt_eval_count}, выход {response.eval_count}")
    print(f"Последовательно за сутки (теория): {86400 / seconds:.0f} документов")
    print(f"Для 10 000 в сутки (теория): {10000 * seconds / 86400:.1f} обработчиков")


if __name__ == "__main__":
    main()
