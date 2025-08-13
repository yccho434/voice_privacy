# app/utils/normalize.py
from __future__ import annotations
import re, unicodedata

# 유니코드 confusables(대표 몇 개) → ASCII
_CONFUSABLES = {
    "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
    "－":"-","—":"-","–":"-","―":"-","-":"-",
    "．":".","·":".","∙":".","。":".",
    "（":"(","）":")","［":"[","］":"]","｛":"{","｝":"}",
    "＠":"@","：":":","／":"/","＼":"\\","，":",",
}

_ZWSP = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width spaces
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

def normalize_text(s: str) -> str:
    # 1) 유니코드 정규화
    s = unicodedata.normalize("NFKC", s)
    # 2) confusables 치환
    s = "".join(_CONFUSABLES.get(ch, ch) for ch in s)
    # 3) zero-width 제거
    s = _ZWSP.sub("", s)
    # 4) 특수 구분자 '.'/':' 통일(시간/전화에서 흔함) - 과도한 치환은 피함
    s = s.replace(" . ", ".").replace(" : ", ":")
    # 5) 연속 공백 정리
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s
