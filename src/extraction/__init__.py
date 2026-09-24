"""Rule-based structured extraction from clinical notes."""
from __future__ import annotations

from .extractor import FIELDS, extract_clinical_data

__all__ = ["FIELDS", "extract_clinical_data"]
