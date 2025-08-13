# utils/spans.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str
    source: str
    confidence: float

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)
