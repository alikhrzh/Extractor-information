"""Small checks against the original document text."""

import re


BIN_RE = re.compile(
    r"(?<!\w)(?:БСН\s*/\s*БИН|БИН\s*/\s*БСН|БИН|БСН)(?!\w)"
    r"\s*[:№-]?\s*([0-9][0-9A-Za-zА-Яа-яЁё]{9,13})(?!\w)",
    re.IGNORECASE,
)


def extract_bin(text: str) -> tuple[str | None, float, str | None]:
    """Return (BIN, confidence, warning) from an explicitly labelled candidate."""
    matches = [match.group(1) for match in BIN_RE.finditer(text)]
    if not matches:
        return None, 0.0, None

    if len(set(matches)) > 1:
        return None, 0.0, "Найдено несколько разных кандидатов БИН/БСН; поставщик не определён."

    candidate = matches[0]
    if re.fullmatch(r"[0-9]{12}", candidate):
        return candidate, 1.0, None
    return None, 0.0, f"БИН/БСН '{candidate}' содержит не 12 цифр; значение не исправлено."


def name_in_text(name: str, text: str) -> bool:
    """Check each original-language part of a model-generated name."""
    source = " ".join(text.casefold().split())
    parts = [" ".join(part.casefold().split()) for part in name.split(" / ")]
    return all(part and part in source for part in parts)
