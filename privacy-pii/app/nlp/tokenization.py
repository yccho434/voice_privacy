# app/nlp/tokenization.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import re
from kiwipiepy import Kiwi
from transformers import AutoTokenizer

_KOREAN_SURNAME = r"(김|이|박|최|정|조|강|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍|고|문|손|양|배|백|허|남|심|노|하|곽|성|차|주|우|구|민|유|나)"
_NAME_CORE = r"[가-힣]{1,3}"
_NAME_PAT = re.compile(rf"\b{_KOREAN_SURNAME}{_NAME_CORE}\b")

@dataclass
class Morpheme:
    text: str
    pos: str
    start: int
    end: int

@dataclass
class Subword:
    text: str
    start: int
    end: int
    id: int

@dataclass
class HybridTokens:
    morphemes: List[Morpheme]
    subwords: List[Subword]
    text: str

class HybridTokenizer:
    def __init__(self, hf_model_name_or_path: str, hf_cache_dir: Optional[str] = None):
        self.kiwi = Kiwi()
        self.kiwi.prepare()
        self.hf_tok = AutoTokenizer.from_pretrained(
            hf_model_name_or_path, cache_dir=hf_cache_dir, use_fast=True
        )

    def morph_tokenize(self, text: str) -> List[Morpheme]:
        out: List[Morpheme] = []
        for tok in self.kiwi.tokenize(text, normalize_coda=True):
            out.append(Morpheme(text=tok.form, pos=tok.tag, start=tok.start, end=tok.end))
        return out

    def subword_tokenize(self, text: str) -> List[Subword]:
        enc = self.hf_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        sub: List[Subword] = []
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if s == e: continue
            piece = text[s:e]
            sub.append(Subword(text=piece, start=s, end=e, id=i))
        return sub

    def analyze(self, text: str) -> HybridTokens:
        return HybridTokens(
            morphemes=self.morph_tokenize(text),
            subwords=self.subword_tokenize(text),
            text=text
        )

    def find_korean_name_spans(self, ht: HybridTokens) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        buf: List[Morpheme] = []
        for m in ht.morphemes:
            if m.pos.startswith("NNP"):
                buf.append(m)
            else:
                if buf:
                    spans.extend(self._flush_name_buf(buf, ht.text)); buf = []
        if buf:
            spans.extend(self._flush_name_buf(buf, ht.text)); buf = []

        for m in _NAME_PAT.finditer(ht.text):
            s, e = m.span()
            spans.append({"start": s, "end": e, "text": ht.text[s:e]})

        spans = self._dedup_spans(spans)
        return spans

    def _flush_name_buf(self, buf: List[Morpheme], text: str) -> List[Dict[str, Any]]:
        s = buf[0].start; e = buf[-1].end
        surface = text[s:e]
        if re.search(r"[^가-힣·\-\s]", surface):
            return []
        comp = re.sub(r"[·\s\-]", "", surface)
        if 2 <= len(comp) <= 4:
            return [{"start": s, "end": e, "text": surface}]
        return []

    @staticmethod
    def _dedup_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not spans: return []
        spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        kept = []; last_end = -1
        for sp in spans:
            if sp["start"] >= last_end:
                kept.append(sp); last_end = sp["end"]
        return kept
