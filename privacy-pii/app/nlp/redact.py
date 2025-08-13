# app/nlp/redact.py
from __future__ import annotations
import re

EMAIL = re.compile(r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+\-])")
URL   = re.compile(r"(?:https?://|www\.)\S+")
DIGIT4 = re.compile(r"\d{4,}")
KNAME = re.compile(r"(김|이|박|최|정|조|강|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍|고|문|손|양|배|백|허|남|심|노|하)[가-힣]{1,2}")
PHONE = re.compile(r"(?<!\d)(?:0(?:2|[3-6][1-5]|70|50|1[016789]))[\s\.-]?\d{3,4}[\s\.-]?\d{4}(?!\d)")
RRN   = re.compile(r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)")
CARD  = re.compile(r"(?<!\d)(?:\d{4}[\s\-]){3}\d{4}(?!\d)")

def _mask_same_len(s: str, ch: str = "X") -> str:
    return "".join(ch if c.isalnum() else c for c in s)

def redact_length_preserving(text: str) -> str:
    out = text
    for pat in [PHONE, RRN, CARD, EMAIL, URL, KNAME, DIGIT4]:
        out = pat.sub(lambda m: _mask_same_len(m.group(0), "X"), out)
    return out
