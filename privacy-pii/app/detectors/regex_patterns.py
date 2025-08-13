# app/detectors/regex_patterns.py
from __future__ import annotations
import os
import re
from typing import List, Dict, Tuple

# 검증기(Quick wins)
from .validators import luhn_ok, rrn_ok, biz_ok, email_domain_allowed

# .env 기반 이메일 도메인 허용/차단(선택)
ALLOW = set(filter(None, (os.getenv("EMAIL_ALLOW", "").lower().split(","))))
DENY  = set(filter(None, (os.getenv("EMAIL_DENY", "").lower().split(","))))

# ------------------------------------------------------------
# 정규식 패턴 (사전 컴파일)
# 라벨은 표준화(번호/계정/금융/신원/주소/URL 등)된 값으로 직접 출력
# ------------------------------------------------------------

# 전화번호 (모바일/지역번호/070/050 등) - 구분자 -, ., 공백 허용
PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:
        0(?:
            2|                    # 02 (서울)
            [3-6][1-5]|           # 031~065 등 지역
            70|50|                # 070/050 (인터넷전화)
            1[016789]             # 010/011/016/017/018/019
        )
    )
    [\s\.-]? \d{3,4} [\s\.-]? \d{4}
    (?!\d)
    """,
    re.X,
)

# 주민등록번호 (단순 패턴) 6-7 or 13자리
RRN_RE = re.compile(r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)")

# 사업자등록번호 3-2-5
BIZ_REG_RE = re.compile(r"(?<!\d)\d{3}[-\s]?\d{2}[-\s]?\d{5}(?!\d)")

# 여권번호 (대한민국 여권 예: M12345678, S12345678 유형 커버)
PASSPORT_RE = re.compile(r"\b[MS]\d{8}\b", re.I)

# 이메일 (태그/서브도메인/국가도메인)
EMAIL_RE = re.compile(
    r"""
    (?<![A-Za-z0-9._%+\-])
    [A-Za-z0-9._%+\-]+
    @
    [A-Za-z0-9.-]+
    \.[A-Za-z]{2,}
    (?![A-Za-z0-9._%+\-])
    """,
    re.X,
)

# URL (http/https, www, 내부망 호스트 포함)
URL_RE = re.compile(
    r"""
    (?:
        (?:https?://|www\.)
        [^\s<>"']+
        |
        \b
        (?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}
        (?:/[^\s<>"']*)?
    )
    """,
    re.X,
)

# IPv4 (내부망 포함) - URL 하위로도 쓰지만 독립 탐지
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b")

# 신용카드 4-4-4-4 (공백/하이픈)
CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[\s\-]){3}\d{4}(?!\d)")

# 계좌번호: 은행 키워드 주변 + 숫자 그룹 2~5개
BANK_WORDS = r"(?:은행|농협|국민|신한|우리|하나|기업|카카오|토스|SC제일|부산|대구|경남|광주|전북|수협|새마을)"
ACCOUNT_HARD_RE = re.compile(
    r"""
    (?:
        (?:%(BANK)s|계좌|입금|출금)\s*[:\(\)]?\s*
    )?
    (?:
        \d{2,6} (?:[\s\-]\d{1,6}){2,5}   # 3~6그룹의 숫자 (한국형 계좌 포맷 다양성)
    )
    """ % {"BANK": BANK_WORDS},
    re.X,
)

# 우편번호 (신규 5자리, 괄호 포함 가능)
ZIP_RE = re.compile(r"(?:\(|\[)?(?P<zip>\d{5})(?:\)|\])?")

# 주소 (도로명/지번 혼합 커버, 괄호 우편번호 허용)
ADDR_RE = re.compile(
    r"""
    (?P<addr>
        (?:[가-힣]{2,}(?:특별시|광역시|특별자치시|도|시))        # 시/도
        \s*
        (?:(?:[가-힣]{1,20})\s*(?:군|구))?                       # 선택: 군/구
        \s*
        (?:(?:[가-힣]{1,20})\s*(?:읍|면|동|가|리))?               # 선택: 읍/면/동/가/리
        \s*
        (?:(?:[가-힣A-Za-z0-9]{1,30})\s*(?:로|길))?              # 선택: 도로명
        \s*
        (?:\d{1,4}(?:-\d{1,4})?)?                                # 선택: 번지
        (?:\s*,?\s*\d{1,4}(?:호|층|동|관|호관)?)?                 # 선택: 호/층/동
        (?:\s*\((?:\d{5})\))?                                    # 선택: (우편번호)
    )
    """,
    re.X,
)

# 날짜/시간 (예시 커버: 2025-08-15, 2024.12.03, 20:30, 09:30)
DATE_RE = re.compile(r"\b(?:\d{4}[-./]\d{1,2}[-./]\d{1,2})\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b")

# ------------------------------------------------------------
# 탐지 로직
# ------------------------------------------------------------
PatternSpec = Tuple[re.Pattern, str, str]  # (pattern, label, subtype)

PATTERNS: List[PatternSpec] = [
    (RRN_RE, "신원", "주민등록번호"),
    (PASSPORT_RE, "신원", "여권번호"),
    (BIZ_REG_RE, "번호", "사업자등록번호"),
    (CARD_RE, "금융", "카드번호"),
    (ACCOUNT_HARD_RE, "금융", "계좌번호"),
    (PHONE_RE, "번호", "전화번호"),
    (EMAIL_RE, "계정", "이메일"),
    (IPV4_RE, "URL", "IP"),
    (URL_RE, "URL", "URL"),
    (ZIP_RE, "주소", "우편번호"),
    (ADDR_RE, "주소", "주소"),
    (DATE_RE, "신원", "날짜"),
    (TIME_RE, "신원", "시간"),
]

def _dedup_and_prune(spans: List[Dict]) -> List[Dict]:
    """겹치는 스팬 정리: 완전 포함되는 짧은 스팬 제거, 동일 범위/텍스트 중복 제거."""
    if not spans:
        return []
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    pruned: List[Dict] = []
    for s in spans:
        overlap = False
        for p in pruned:
            if s["start"] >= p["start"] and s["end"] <= p["end"]:
                overlap = True; break
            if s["start"] == p["start"] and s["end"] == p["end"] and s["label"] == p["label"]:
                overlap = True; break
        if not overlap:
            pruned.append(s)
    pruned.sort(key=lambda s: (s["start"], s["end"]))
    return pruned

def detect_by_regex(text: str) -> List[Dict]:
    """정규식 기반 1차 탐지 + 검증기(Quick wins).
    반환: [{label, subtype, text, start, end}]
    """
    hits: List[Dict] = []
    for pattern, label, subtype in PATTERNS:
        for m in pattern.finditer(text):
            span_text = m.group("addr") if "addr" in pattern.groupindex else m.group(0)
            start = m.start("addr") if "addr" in pattern.groupindex else m.start()
            end = start + len(span_text)

            # URL 패턴이 이메일을 과포함하는 경우 방어
            if label == "URL":
                if EMAIL_RE.fullmatch(span_text or ""):
                    continue

            # 계좌번호: 너무 광범위한 숫자 그룹 오탐 방지(하드 필터)
            if label == "금융" and subtype == "계좌번호":
                raw = span_text
                sep_cnt = len(re.findall(r"[\s\-]", raw))
                bank_nearby = False
                left = max(0, start - 15)
                ctx = text[left:start]
                if re.search(BANK_WORDS, ctx):
                    bank_nearby = True
                if sep_cnt < 2 and not bank_nearby:
                    continue

            # ---------- Quick wins: 검증기 ----------
            if label == "금융" and subtype == "카드번호":
                if not luhn_ok(span_text):
                    continue
            if label == "신원" and subtype == "주민등록번호":
                if not rrn_ok(span_text):
                    continue
            if label == "번호" and subtype == "사업자등록번호":
                if not biz_ok(span_text):
                    continue
            if label == "계정" and subtype == "이메일":
                if not email_domain_allowed(span_text, ALLOW or None, DENY or None):
                    continue
            # --------------------------------------

            hits.append({
                "label": label,
                "subtype": subtype,
                "text": span_text,
                "start": start,
                "end": end,
            })

    return _dedup_and_prune(hits)
