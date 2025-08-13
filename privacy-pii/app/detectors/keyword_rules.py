# app/detectors/keyword_rules.py
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ---- 기존 번호 세분화 키워드 ----
KEYWORDS: Dict[str, List[str]] = {
    "전화번호": ["전화", "휴대폰", "핸드폰", "연락처", "콜", "통화", "대표번호", "번호로", "문자"],
    "계좌번호": ["계좌", "입금", "송금", "이체", "은행", "우리은행", "국민", "신한", "하나", "농협", "기업", "수취", "예금주"],
    "우편번호": ["우편", "ZIP", "zipcode", "우편번호", "우체국"],
    "여권번호": ["여권"],
    "운전면허": ["운전면허", "면허증"],
    "IP": ["IP", "아이피", "서버", "내부망", "사설IP", "공인IP", "내부", "외부망"],
}

# ---- 상담 도메인 라벨 키워드 (추가 스팬용) ----
DOMAIN_KEYWORDS = {
    "정신건강": ["우울","불안","공황","ADHD","PTSD","양극성","조현","진단","약물","복용","트라우마","외상"],
    "상담이력": ["상담","세션","회기","내담자","초진","재진","기록","케이스노트"],
    "생활패턴": ["수면","불면","과수면","악몽","음주","흡연","운동","식습관","게임중독"],
    "사건경험": ["학대","폭력","따돌림","사고","상해","성폭력","자살","시도","가정폭력","중독"],
}

NEAR = 24

RE_PHONE = re.compile(r"(?:^|[^0-9])(01[016789])[-.\s]?\d{3,4}[-.\s]?\d{4}(?:$|[^0-9])")
RE_IP    = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_ZIP   = re.compile(r"\b\d{5}\b")

def _score_context(text: str, s: int, e: int) -> Tuple[Optional[str], float, Optional[str]]:
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

def _scan_domain_spans(text: str) -> List[dict]:
    hits: List[dict] = []
    for lab, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            for m in re.finditer(re.escape(kw), text):
                s, e = m.start(), m.end()
                hits.append({
                    "start": s, "end": e, "text": text[s:e],
                    "label": lab, "label_adjusted": lab, "subtype": None,
                    "score": 0.6, "source": "keyword-domain"
                })
    return hits

def detect_by_keywords(text: str, rx_hits: List[dict]) -> List[dict]:
    out: List[dict] = []
    for s in rx_hits:
        lab = s.get("label", "")
        ladj = s.get("label_adjusted")
        st = s["start"]; en = s["end"]
        new = dict(s)

        if ladj:
            out.append(new)
            continue

        if lab == "번호":
            snippet = text[st:en]
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
                sub, sc, kw = _score_context(text, st, en)
                if sub:
                    new["subtype"] = sub
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

    # --- 도메인 키워드 기반 추가 스팬 ---
    out.extend(_scan_domain_spans(text))
    return out
