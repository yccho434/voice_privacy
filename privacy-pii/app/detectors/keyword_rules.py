# app/detectors/keyword_rules.py
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 문맥 키워드: 모호한 '번호' 라벨을 전화/계좌/우편번호/IP/여권/면허 등으로 분류
# ──────────────────────────────────────────────────────────────────────────────
KEYWORDS: Dict[str, List[str]] = {
    "전화번호": ["전화", "휴대폰", "핸드폰", "연락처", "콜", "통화", "대표번호", "번호로", "문자"],
    "계좌번호": ["계좌", "입금", "송금", "이체", "은행", "우리은행", "국민", "신한", "하나", "농협", "기업", "수취", "예금주"],
    "우편번호": ["우편", "ZIP", "zipcode", "우편번호", "우체국"],
    "여권번호": ["여권"],
    "운전면허": ["운전면허", "면허증"],
    "IP": ["IP", "아이피", "서버", "내부망", "사설IP", "공인IP", "내부", "외부망"],
}

NEAR = 24  # 문맥 반경(문자 수)

# 간단 휴리스틱 패턴 (정규식으로 힌트)
RE_PHONE = re.compile(r"(?:^|[^0-9])(01[016789])[-.\s]?\d{3,4}[-.\s]?\d{4}(?:$|[^0-9])")
RE_IP    = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_ZIP   = re.compile(r"\b\d{5}\b")

def _score_context(text: str, s: int, e: int) -> Tuple[Optional[str], float, Optional[str]]:
    """
    '번호' 라벨에 대해 주변 문맥으로 세부유형(subtype) 추정.
    반환: (subtype, score, keyword)
    """
    left = max(0, s - NEAR)
    right = min(len(text), e + NEAR)
    ctx = text[left:right]
    cnt = Counter()
    hit_kw = None
    for sub, kws in KEYWORDS.items():
        for k in kws:
            if k in ctx:
                cnt[sub] += 1
                if hit_kw is None:
                    hit_kw = k
    if not cnt:
        return None, 0.0, None
    subtype, hits = cnt.most_common(1)[0]
    score = min(1.0, 0.6 + 0.1 * hits)
    return subtype, score, hit_kw

def detect_by_keywords(text: str, rx_hits: List[dict]) -> List[dict]:
    """
    정규식 결과(rx_hits)를 받아 모호한 라벨을 문맥으로 보정.
    이미 label_adjusted가 있는 경우는 절대 덮어쓰지 않음.
    """
    out: List[dict] = []
    for s in rx_hits:
        lab = s.get("label", "")
        ladj = s.get("label_adjusted")  # 이미 확정된 라벨이면 존중
        st = s["start"]; en = s["end"]
        new = dict(s)

        # 이미 확정 라벨이면 그대로
        if ladj:
            out.append(new)
            continue

        if lab == "번호":
            snippet = text[st:en]

            # 1) 휴리스틱 우선
            if RE_PHONE.search(text[max(0, st - 2):min(len(text), en + 2)]):
                new["subtype"] = "전화번호"
                new["label_adjusted"] = "번호"
                new["score"] = max(new.get("score", 0), 0.95)
            elif RE_IP.search(snippet):
                new["subtype"] = "IP"
                new["label_adjusted"] = "번호"
                new["score"] = max(new.get("score", 0), 0.90)
            elif RE_ZIP.fullmatch(snippet.replace(" ", "")):
                new["subtype"] = "우편번호"
                new["label_adjusted"] = "번호"
                new["score"] = max(new.get("score", 0), 0.85)
            else:
                # 2) 문맥 기반 분류
                sub, sc, kw = _score_context(text, st, en)
                if sub:
                    new["subtype"] = sub
                    # 계좌번호는 금융으로 승격
                    if sub == "계좌번호":
                        new["label_adjusted"] = "금융"
                    else:
                        new["label_adjusted"] = "번호"
                    new["score"] = max(new.get("score", 0), sc)
                    if kw:
                        new["keyword"] = kw
                else:
                    new["label_adjusted"] = "번호"

        elif lab in ("계정", "URL", "금융", "주소", "이름", "소속", "신원"):
            new.setdefault("label_adjusted", lab)
        else:
            new.setdefault("label_adjusted", lab or "신원")

        out.append(new)
    return out
