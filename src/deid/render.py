"""Rendering of redacted note text.

Byte-for-byte the same operation the evaluator performs when it checks
`deidentified_text`: spans are replaced in reverse order so that earlier
offsets stay valid. Kept as its own module so a test can assert equality with
`evaluator.evaluate.render_deidentified` directly.
"""
from __future__ import annotations

from typing import Any


def render_deidentified(note: str, spans: list[dict[str, Any]]) -> str:
    output = note
    for span in reversed(spans):
        output = output[: span["start"]] + f"[{span['label']}]" + output[span["end"] :]
    return output
