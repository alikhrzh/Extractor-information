"""Small checks against the original document text."""

import re

from src.models import Document


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


def apply_validators(doc: Document, text: str) -> tuple[Document, list[str]]:
    """Apply only BIN and name checks; report every changed value separately."""
    data = doc.model_dump()
    corrections = []
    bin_value, confidence, warning = extract_bin(text)

    # A missing match despite a BIN label may mean an unsupported layout.
    ambiguous = warning is not None and "несколько разных" in warning
    has_label = "БИН" in text.upper() or "БСН" in text.upper()
    if not ambiguous and (bin_value is not None or warning is not None or not has_label):
        if data["supplier_bin"] != bin_value:
            corrections.append(f"supplier_bin: {data['supplier_bin']!r} -> {bin_value!r}")
        data["supplier_bin"] = bin_value
        if data["field_confidence"]["supplier_bin"] != confidence:
            corrections.append(f"field_confidence.supplier_bin: -> {confidence}")
        data["field_confidence"]["supplier_bin"] = confidence
        if warning and warning not in data["warnings"]:
            data["warnings"].append(warning)
            corrections.append(f"warnings: added {warning}")

    def keep_source_parts(name: str) -> str | None:
        parts = [part.strip() for part in name.split(" / ")]
        present = [part for part in parts if name_in_text(part, text)]
        return " / ".join(present) or None

    if data["supplier_name"] and not name_in_text(data["supplier_name"], text):
        new_name = keep_source_parts(data["supplier_name"])
        corrections.append(f"supplier_name: {data['supplier_name']!r} -> {new_name!r}")
        data["supplier_name"] = new_name

    kept_items = []
    for item in data["line_items"]:
        if not name_in_text(item["name"], text):
            new_name = keep_source_parts(item["name"])
            corrections.append(f"line_item.name: {item['name']!r} -> {new_name!r}")
            if new_name is None:
                continue
            item["name"] = new_name
        kept_items.append(item)
    data["line_items"] = kept_items

    return Document.model_validate(data), corrections
