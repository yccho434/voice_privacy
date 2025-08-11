# app/test.py
from __future__ import annotations

import os
import re
import json
from typing import List, Dict

import streamlit as st

# 내부 모듈
from detectors.regex_patterns import detect_by_regex
from detectors.keyword_rules import detect_by_keywords

# ──────────────────────────────────────────────────────────────────────────────
# (선택) NER 로더: .env(HF_NER_MODEL, HF_CACHE_DIR) 있으면 사용, 없으면 비활성
# ──────────────────────────────────────────────────────────────────────────────
NER_AVAILABLE = False
NER_PIPE = None
try:
    HF_MODEL = os.getenv("HF_NER_MODEL", "").strip()
    HF_CACHE = os.getenv("HF_CACHE_DIR", "").strip() or None
    if HF_MODEL:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        tok = AutoTokenizer.from_pretrained(HF_MODEL, cache_dir=HF_CACHE)
        mdl = AutoModelForTokenClassification.from_pretrained(HF_MODEL, cache_dir=HF_CACHE)
        NER_PIPE = pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")
        NER_AVAILABLE = True
except Exception as e:
    # 로드 실패는 무시(옵션)
    NER_AVAILABLE = False
    NER_PIPE = None

# ──────────────────────────────────────────────────────────────────────────────
# 간단 형태소 기반(룰) 이름 후보 탐지기: 고유명(성씨+1~2자)
# ──────────────────────────────────────────────────────────────────────────────
SURNAME = "김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허남심노하"
NAME_RE = re.compile(rf"([{SURNAME}])[가-힣]{{1,2}}(씨|님)?")

def detect_names_by_morph(text: str) -> List[dict]:
    hits: List[dict] = []
    for m in NAME_RE.finditer(text):
        s, e = m.span()
        # 앞뒤에 조사/숫자/이메일 문맥이면 제외하는 간단한 필터
        left = text[s-1:s] if s > 0 else ""
        right = text[e:e+1] if e < len(text) else ""
        if left and left.isalnum():  # 영문/숫자 붙으면 제외
            continue
        if right and right.isalnum():
            continue
        hits.append({
            "start": s, "end": e,
            "label": "이름", "label_adjusted": "이름",
            "subtype": "고유명(형태소)", "score": 0.88, "source": "morph"
        })
    return hits

# ──────────────────────────────────────────────────────────────────────────────
# 겹침/우선순위 병합
# ──────────────────────────────────────────────────────────────────────────────
LABEL_PRIORITY = {
    ("번호", "주민등록번호"): 99,
    ("금융", "계좌번호"): 100,
    ("계정", "이메일"): 95,
    ("URL", None): 92,
    ("주소", None): 93,
    ("번호", "전화번호"): 90,
    ("번호", "여권번호"): 88,
    ("번호", "운전면허"): 88,
    ("번호", "IP"): 85,
    ("이름", None): 80,
    ("소속", None): 70,
    ("신원", None): 60,
}

def _prio(label: str, subtype: str | None) -> int:
    key = (label, (subtype or None))
    if key in LABEL_PRIORITY:
        return LABEL_PRIORITY[key]
    for (l, s), v in LABEL_PRIORITY.items():
        if l == label and s is None:
            return v
    return 50

def merge_with_priority(spans: List[dict]) -> List[dict]:
    """
    겹치는 스팬들을 우선순위(라벨/서브타입) → 길이(긴 것) → 앞선 것 기준으로 선택
    """
    normalized: List[dict] = []
    for s in spans:
        s = dict(s)
        s.setdefault("label_adjusted", s.get("label", ""))
        s.setdefault("subtype", None)
        s["_priority"] = _prio(s["label_adjusted"] or s["label"], s.get("subtype"))
        normalized.append(s)

    normalized.sort(key=lambda s: (s["start"], -s["_priority"], -(s["end"] - s["start"])))

    chosen: List[dict] = []
    for cand in normalized:
        keep = True
        to_remove = []
        for sel in chosen:
            if not (cand["end"] <= sel["start"] or cand["start"] >= sel["end"]):
                # 겹침
                if cand["_priority"] > sel["_priority"]:
                    to_remove.append(sel)
                else:
                    keep = False
                    break
        if keep:
            for rm in to_remove:
                chosen.remove(rm)
            # cand와 다시 겹치지 않는지 확인해 중복 제거
            chosen = [s for s in chosen if (s["end"] <= cand["start"] or s["start"] >= cand["end"])]
            chosen.append(cand)

    chosen.sort(key=lambda s: s["start"])
    for s in chosen:
        s.pop("_priority", None)
    return chosen

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="한국어 개인정보 탐지 (정규식 + 키워드 + NER + 형태소)", layout="wide")

st.title("한국어 개인정보 탐지 (정규식 + 키워드 + NER + 형태소)")
st.caption("정규식 → 키워드 문맥 → (선택) NER → 형태소 이름 후보 → 우선순위 병합")

default_text = (
    "홍길동(32세)입니다. 연락처 010-1234-5678, 이메일 hong@example.com, "
    "주소 서울시 마포구 상암동 123-45. 계좌 1002-345-678901(우리은행). "
    "주민등록번호 801010-1234567. [상담사] 20:30에 불안 지수 7/10 올라옴. "
    "박지영 씨, 2025-08-15 14:30 일정. 내부망 https://intra.mycompany.co.kr/meeting. "
    "센터 대표번호 02-777-0000."
)
text = st.text_area("텍스트 입력", value=default_text, height=220)

colA, colB, colC = st.columns([1,1,1])
use_ner = colA.checkbox("NER 사용(옵션)", value=NER_AVAILABLE)
colB.write(f"NER 로드 상태: {'OK' if NER_AVAILABLE else '미사용'}")
run = st.button("실행하기")

def _render_highlight(text: str, spans: List[dict]):
    """
    간단 하이라이트 렌더러 (배경색은 라벨별 고정 팔레트)
    """
    palette = {
        "금융": "#4CAF50",
        "계정": "#607D8B",
        "URL": "#8E24AA",
        "번호": "#3F51B5",
        "주소": "#FFC107",
        "이름": "#EF5350",
        "소속": "#009688",
        "신원": "#90A4AE",
    }
    st.subheader("하이라이트 미리보기")
    out = []
    last = 0
    spans = sorted(spans, key=lambda s: s["start"])
    for s in spans:
        out.append(text[last:s["start"]])
        label = s.get("label_adjusted") or s.get("label")
        color = palette.get(label, "#BDBDBD")
        chunk = text[s["start"]:s["end"]]
        tip = f'{label} / {s.get("subtype") or ""}'.strip()
        out.append(f'<span style="background:{color}; color:#111; padding:2px 4px; border-radius:4px;" title="{tip}">{chunk}</span>')
        last = s["end"]
    out.append(text[last:])
    st.markdown("".join(out), unsafe_allow_html=True)

if run:
    # 1) 정규식
    rx_hits = detect_by_regex(text)

    # 2) 키워드 보정
    kw_hits = detect_by_keywords(text, rx_hits)

    # 3) (옵션) NER
    ner_hits: List[dict] = []
    if use_ner and NER_PIPE is not None:
        try:
            res = NER_PIPE(text)
            for r in res:
                ner_hits.append({
                    "start": r["start"], "end": r["end"],
                    "label": r["entity_group"], "label_adjusted": r["entity_group"],
                    "subtype": None, "score": float(r.get("score", 0.0)), "source": "ner"
                })
        except Exception:
            ner_hits = []

    # 4) 형태소 기반 이름 후보(룰)
    morph_name_hits = detect_names_by_morph(text)

    # 5) 스팬 합치기
    spans: List[dict] = []
    for r in kw_hits:
        spans.append({
            "start": r["start"], "end": r["end"],
            "label": r.get("label", ""), "label_adjusted": r.get("label_adjusted", r.get("label","")),
            "subtype": (r.get("subtype") or "").strip() or None,
            "score": r.get("score", 0) or 0,
            "source": r.get("source","regex")
        })
    for n in ner_hits:
        spans.append({
            "start": n["start"], "end": n["end"],
            "label": n["label"], "label_adjusted": n["label"],
            "subtype": (n.get("subtype") or "").strip() or None,
            "score": n.get("score", 0) or 0,
            "source": "ner"
        })
    for m in morph_name_hits:
        spans.append({
            "start": m["start"], "end": m["end"],
            "label": m["label_adjusted"], "label_adjusted": m["label_adjusted"],
            "subtype": (m.get("subtype") or "고유명(형태소)").strip(),
            "score": m.get("score", 0.9),
            "source": "morph"
        })

    # 6) 우선순위 병합
    merged = merge_with_priority(spans)

    # 결과 표시
    _render_highlight(text, merged)

    with st.expander("디버그: 원시 정규식 히트"):
        st.json(rx_hits)
    with st.expander("디버그: 키워드 보정 후"):
        st.json(kw_hits)
    with st.expander("디버그: NER 히트"):
        st.json(ner_hits)
    with st.expander("디버그: 형태소 이름 히트"):
        st.json(morph_name_hits)
    with st.expander("최종 병합 스팬"):
        st.json(merged)
