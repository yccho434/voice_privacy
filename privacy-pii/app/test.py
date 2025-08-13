# app/test.py
from __future__ import annotations

import os
import re
from typing import List, Dict
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # .env: HF_NER_MODEL, HF_CACHE_DIR, OPENAI_API_KEY 등

from detectors.regex_patterns import detect_by_regex
from detectors.keyword_rules import detect_by_keywords
from nlp.gpt_post import validate_with_gpt
from utils.normalize import normalize_text

# =========================
# 리소스 로더
# =========================
def get_gpt_ready() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

@st.cache_resource(show_spinner=False)
def make_ner_pipe():
    try:
        HF_MODEL = os.getenv("HF_NER_MODEL", "").strip()
        HF_CACHE = os.getenv("HF_CACHE_DIR", "").strip() or None
        if not HF_MODEL:
            return None
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        tok = AutoTokenizer.from_pretrained(HF_MODEL, cache_dir=HF_CACHE)
        mdl = AutoModelForTokenClassification.from_pretrained(HF_MODEL, cache_dir=HF_CACHE)
        return pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")
    except Exception:
        return None

NER_PIPE = make_ner_pipe()
NER_AVAILABLE = NER_PIPE is not None
GPT_READY = get_gpt_ready()

# =========================
# 이름 휴리스틱(확장)
# =========================
SURNAMES = [
    "남궁","제갈","선우","사공","서문","독고","황보","동방","어금","망절","장곡","남평","탁","공","백",
    "김","이","박","최","정","조","강","윤","장","임","한","오","서","신","권","황","안","송","류","홍",
    "전","고","문","양","손","배","백","허","남","심","노","하","곽","성","차","주","우","구","민","유","나"
]
REL_TITLES = {"엄마","아빠","부모님","선생님","상담사","센터장","팀장","과장","대표","원장","교수","코치"}

NAME_RE = re.compile(rf"\b({'|'.join(map(re.escape, SURNAMES))})[·\-\s]?[가-힣]{{1,2}}(씨|님)?\b")

def detect_names_by_morph(text: str) -> List[dict]:
    hits: List[dict] = []
    for m in NAME_RE.finditer(text):
        s, e = m.span()
        surf = text[s:e]
        left = text[s-1:s] if s > 0 else ""
        right = text[e:e+1] if e < len(text) else ""
        if left and left.isalnum():
            continue
        if right and right.isalnum():
            continue
        if surf in REL_TITLES:
            continue
        hits.append({
            "start": s, "end": e,
            "label": "이름", "label_adjusted": "이름",
            "subtype": "고유명(형태소+복성)", "score": 0.9, "source": "morph",
            "text": surf,
        })
    return hits

# =========================
# 병합(우선순위 + IoU/score)
# =========================
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
    # 상담 도메인
    ("정신건강", None): 65,
    ("상담이력", None): 55,
    ("생활패턴", None): 50,
    ("사건경험", None): 68,
}
def _prio(label: str, subtype: str | None) -> int:
    key = (label, (subtype or None))
    if key in LABEL_PRIORITY:
        return LABEL_PRIORITY[key]
    for (l, s), v in LABEL_PRIORITY.items():
        if l == label and s is None:
            return v
    return 50

def _iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union else 0.0

def merge_with_priority(spans: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for s in spans:
        s = dict(s)
        s.setdefault("label_adjusted", s.get("label", ""))
        s.setdefault("subtype", None)
        s.setdefault("score", 0.0)
        s["_priority"] = _prio(s["label_adjusted"] or s["label"], s.get("subtype"))
        normalized.append(s)

    normalized.sort(key=lambda s: (s["start"], -s["_priority"], -(s["end"] - s["start"])))
    chosen: List[dict] = []
    for cand in normalized:
        keep = True
        to_remove = []
        for sel in chosen:
            if not (cand["end"] <= sel["start"] or cand["start"] >= sel["end"]):
                if (cand.get("label_adjusted")==sel.get("label_adjusted")) and _iou(cand, sel) >= 0.7:
                    if cand.get("score",0) > sel.get("score",0):
                        to_remove.append(sel)
                    else:
                        keep = False
                elif cand["_priority"] > sel["_priority"]:
                    to_remove.append(sel)
                else:
                    keep = False
        if keep:
            for rm in to_remove:
                chosen.remove(rm)
            chosen = [s for s in chosen if (s["end"] <= cand["start"] or s["start"] >= cand["end"])]
            chosen.append(cand)

    chosen.sort(key=lambda s: s["start"])
    for s in chosen:
        s.pop("_priority", None)
    return chosen

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="한국어 PII 탐지 (정규식+키워드+NER+형태소+GPT)", layout="wide")

if "input_text" not in st.session_state:
    st.session_state.input_text = (
        "홍길동(32세)입니다. 연락처 010-1234-5678, 이메일 hong@example.com, "
        "주소 서울시 마포구 상암동 123-45. 계좌 1002-345-678901(우리은행). "
        "주민등록번호 801010-1******. [상담사] 20:30에 불안 지수 7/10 올라옴. "
        "박지영 씨, 2025-08-15 14:30 일정. 내부망 https://intra.mycompany.co.kr/meeting. "
        "센터 대표번호 02-777-0000."
    )
if "use_ner" not in st.session_state:
    st.session_state.use_ner = bool(NER_AVAILABLE)
if "use_gpt" not in st.session_state:
    st.session_state.use_gpt = False

st.title("한국어 개인정보 탐지 (정규식 + 키워드 + NER + 형태소 + GPT)")
st.caption("정규식 → 키워드 문맥 → (선택) NER/형태소 → (선택) GPT 문맥 검증/보완 → 우선순위 병합")

text = st.text_area("텍스트 입력", key="input_text", height=220)

colA, colB, colC = st.columns([1,1,2])
use_ner = colA.checkbox("NER 사용(옵션)", key="use_ner", disabled=not NER_AVAILABLE)
use_gpt = colB.checkbox("GPT 문맥 검증/보완(옵션)", key="use_gpt", disabled=not GPT_READY)

ner_status = "OK" if NER_AVAILABLE else "미사용"
gpt_status = "OK" if GPT_READY else "없음"
colC.write(f"NER: {ner_status} / GPT 키: {gpt_status}")

if not NER_AVAILABLE and use_ner:
    st.warning("HF_NER_MODEL이 설정되지 않았거나 모델 로드에 실패하여 NER을 사용할 수 없습니다.")
if not GPT_READY and use_gpt:
    st.warning("OPENAI_API_KEY가 없어서 GPT 검증을 사용할 수 없습니다.")

run = st.button("실행하기", key="run_btn")

# =========================
# 렌더링 & 요약
# =========================
def _render_highlight(text_src: str, spans: List[dict]):
    palette = {
        "금융": "#4CAF50","계정": "#607D8B","URL": "#8E24AA","번호": "#3F51B5",
        "주소": "#FFC107","이름": "#EF5350","소속": "#009688","신원": "#90A4AE",
        "정신건강":"#7E57C2","상담이력":"#26A69A","생활패턴":"#FF7043","사건경험":"#5C6BC0",
    }
    st.subheader("하이라이트 미리보기")
    out = []
    last = 0
    spans = sorted(spans, key=lambda s: s["start"])
    for s in spans:
        out.append(text_src[last:s["start"]])
        label = s.get("label_adjusted") or s.get("label")
        color = palette.get(label, "#BDBDBD")
        chunk = text_src[s["start"]:s["end"]]
        tip = f'{label}{(" / "+s.get("subtype")) if s.get("subtype") else ""}'
        out.append(
            f'<span style="background:{color}33; padding:2px 2px; border-radius:3px;" title="{tip}">{chunk}</span>'
        )
        last = s["end"]
    out.append(text_src[last:])
    st.markdown("".join(out), unsafe_allow_html=True)

def summarize_by_label(spans: List[dict], text_src: str, max_examples: int = 5):
    bucket = defaultdict(list)
    for s in spans:
        lab = (s.get("label_adjusted") or s.get("label") or "").strip()
        if not lab or lab.upper().startswith("LABEL_"):
            continue
        txt = s.get("text") or text_src[s["start"]:s["end"]]
        s = dict(s); s["text"] = txt
        bucket[lab].append(s)
    rows = []
    for lab, items in sorted(bucket.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        seen = set(); samples = []
        for it in items:
            t = (it.get("text") or "").strip()
            if t and t not in seen:
                samples.append(t); seen.add(t)
            if len(samples) >= max_examples: break
        rows.append({"라벨": lab, "개수": len(items), "예시": " | ".join(samples)})
    return rows

# =========================
# 실행 파이프라인
# =========================
if run:
    current_text = normalize_text(st.session_state.input_text)

    # 1) 정규식
    rx_hits = detect_by_regex(current_text)
    # 2) 키워드 보정
    kw_hits = detect_by_keywords(current_text, rx_hits)

    # 3) (옵션) NER
    ner_hits: List[dict] = []
    if use_ner and NER_PIPE is not None:
        try:
            res = NER_PIPE(current_text)
            for r in res:
                lab = r["entity_group"]
                if str(lab).upper().startswith("LABEL_"):
                    continue
                ner_hits.append({
                    "start": r["start"], "end": r["end"],
                    "label": lab, "label_adjusted": lab,
                    "subtype": None, "score": float(r.get("score", 0.0)), "source": "ner",
                    "text": current_text[r["start"]:r["end"]],
                })
        except Exception as e:
            st.warning(f"NER 추론 오류: {e}")
            ner_hits = []

    # 4) 형태소 이름(확장)
    morph_name_hits = detect_names_by_morph(current_text)

    # 5) 스팬 수집
    spans: List[dict] = []
    for r in kw_hits:
        spans.append({
            "start": r["start"], "end": r["end"],
            "label": r.get("label", ""), "label_adjusted": r.get("label_adjusted", r.get("label","")),
            "subtype": (r.get("subtype") or "").strip() or None,
            "score": r.get("score", 0) or 0,
            "source": r.get("source","regex"),
            "text": r.get("text","") or current_text[r["start"]:r["end"]],
        })
    for n in ner_hits:
        spans.append(n)
    for m in morph_name_hits:
        spans.append(m)

    # 6) (옵션) GPT 보강 (Union)
    if use_gpt and GPT_READY:
        spans_after_gpt = validate_with_gpt(current_text, spans)
        if spans_after_gpt:
            spans = spans + spans_after_gpt

    # 7) 최종 병합
    merged = merge_with_priority(spans)

    # 출력
    _render_highlight(current_text, merged)

    st.markdown("---")
    st.subheader("라벨별 태깅 요약")
    import pandas as pd
    summary_rows = summarize_by_label(merged, current_text, max_examples=5)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "CSV 다운로드 (라벨별 요약)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="pii_label_summary.csv",
            mime="text/csv",
        )
    else:
        st.write("태깅 결과가 없습니다.")

    # 디버그
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
