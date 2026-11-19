"""Typed and validated output for one extracted document."""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator


Number = StrictInt | StrictFloat


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: StrictStr = Field(min_length=1)
    qty: Number | None
    price: Number | None


class FieldConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    supplier_bin: float = Field(ge=0, le=1, strict=True)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    doc_type: Literal["invoice", "commercial_offer", "act", "unknown"]
    supplier_name: StrictStr | None
    supplier_bin: StrictStr | None
    doc_number: StrictStr | None
    doc_date: StrictStr | None
    currency: Literal["KZT", "USD", "EUR"] | None
    total_amount: Number | None
    vat_amount: Number | None
    line_items: list[LineItem]
    field_confidence: FieldConfidence
    warnings: list[StrictStr]

    @field_validator("supplier_bin")
    @classmethod
    def validate_bin(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9]{12}", value):
            raise ValueError("supplier_bin must contain exactly 12 digits")
        return value

    @field_validator("doc_date")
    @classmethod
    def validate_date(cls, value: str | None) -> str | None:
        if value is not None:
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                raise ValueError("doc_date must use YYYY-MM-DD")
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("doc_date must be a real calendar date") from exc
        return value
