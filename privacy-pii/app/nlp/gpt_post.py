# app/nlp/gpt_post.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, textwrap, time, math, re

try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _client = None

from .redact import redact_length_preserving

# ────────────────────────────────────────────────────────────────────
# 모델/라벨/우선순위
# ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")

# 표준 라벨 집합과 정규화 맵
LABELS = {
    "이름", "소속", "주소", "신원", "계정", "번호", "금융", "URL",
    "정신건강", "상담이력", "생활패턴", "사건경험"
}
LABEL_ALIASES = {
    "기관": "소속", "회사": "소속", "학교": "소속", "조직": "소속",
    "전화번호": "번호", "휴대폰번호": "번호", "핸드폰": "번호",
    "계좌": "금융", "카드": "금융",
    "웹주소": "URL", "링크": "URL"
}
# 더 특이한/민감한 라벨을 우선
LABEL_PRIORITY = {
    "정신건강": 100, "상담이력": 95, "사건경험": 90, "생활패턴": 85,
    "이름": 80, "신원": 75, "주소": 70, "금융": 65, "계정": 60,
    "번호": 50, "URL": 40, "소속": 35
}

MAX_ADDITION_LEN = 128  # 추가 탐지 스팬 최대 길이

def _normalize_label(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    y = str(x).strip()
    if y in LABELS:
        return y
    y = LABEL_ALIASES.get(y, y)
    return y if y in LABELS else None

# ────────────────────────────────────────────────────────────────────
# 유틸: 텍스트 슬라이싱/윈도우링
# ────────────────────────────────────────────────────────────────────
def _slice(text: str, s: int, e: int, L: int = 120, R: int = 120) -> str:
    return text[max(0, s - L): min(len(text), e + R)]

def _build_items_masked(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for i, sp in enumerate(spans):
        s, e = int(sp["start"]), int(sp["end"])
        if not (0 <= s < e <= len(text)):
            continue
        chunk = text[s:e]
        ctx = _slice(text, s, e)
        items.append({
            "id": str(i),
            "span_text": redact_length_preserving(chunk),
            "proposed_label": sp.get("label_adjusted") or sp.get("label"),
            "proposed_subtype": sp.get("subtype"),
            "context": redact_length_preserving(ctx),
        })
    return items

def _build_windows_masked(text: str, win: int = 600, step: int = 400) -> List[Dict[str, Any]]:
    windows = []
    for i in range(0, len(text), step):
        seg = text[i: i + win]
        if not seg:
            break
        windows.append({
            "id": f"win-{i}",
            "context": redact_length_preserving(seg),
            "offset": i,  # 전역 오프셋
        })
    return windows

# ────────────────────────────────────────────────────────────────────
# 프롬프트 구성 (윈도우 상대 좌표 계약 강화)
# ────────────────────────────────────────────────────────────────────
def _prompt_payload(items: List[Dict[str, Any]],
                    windows: Optional[List[Dict[str, Any]]] = None) -> str:

    labels_list = ", ".join(sorted(LABELS, key=lambda x: -LABEL_PRIORITY.get(x, 0)))

    guide = textwrap.dedent(f"""
        당신은 한국어 텍스트의 개인정보(PII) 후보를 검증/보완하는 심사관입니다.
        표준 라벨(반드시 이 중에서만 선택): {labels_list}

        아래 JSON 스키마로만 답하세요. 다른 텍스트 금지.
        {{
          "decisions": [
            {{"id":"<입력 id>", "keep":true|false, "label":"<표준 라벨 또는 null>", "subtype":"<세부유형 또는 null>", "reason":"<간단 근거>", "confidence": <0~1 숫자>}}
          ],
          "additions": [
            {{
              "win_id": "<윈도우 id>",
              "start_rel": <int>, "end_rel": <int>,
              "label": "<표준 라벨>", "subtype": "<세부유형 또는 null>",
              "reason": "<간단 근거>", "confidence": <0~1 숫자>
            }}
          ]
        }}

        기준:
        - 제품코드·주문번호·일반 식별 불가 정보는 제외.
        - 상담/진단/약물/트라우마/사건/생활 습관은 도메인 라벨(정신건강, 상담이력, 사건경험, 생활패턴)로 분류.
        - 라벨은 반드시 표준 라벨 중 하나만 사용. 모호하면 null 또는 더 일반 라벨(신원/번호 등) 선택.
        - 각 decision/addition에는 근거와 0~1 신뢰도(confidence)를 기입.

        items: 후보 스팬 검증/교정용 (id로 참조)
        windows: 추가 탐지 전용. "start_rel/end_rel"는 해당 win_id context 내부의 상대 좌표.
    """).strip()

    payload: Dict[str, Any] = {"guide": guide, "items": items}
    if windows:
        # 모델이 참조할 수 있도록 최소 정보만 전달
        payload["windows"] = [{"id": w["id"], "context": w["context"]} for w in windows]
        payload["note"] = "반드시 윈도우 상대 좌표(start_rel/end_rel)로 additions를 작성하세요."
    return json.dumps(payload, ensure_ascii=False)

# ────────────────────────────────────────────────────────────────────
# JSON 파싱/검증/정규화
# ────────────────────────────────────────────────────────────────────
def _safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # 관대한 수습: 백틱/코드펜스 제거, 트레일링 텍스트 제거 시도
        s2 = s.strip()
        s2 = re.sub(r"^```(json)?", "", s2, flags=re.I).strip()
        s2 = re.sub(r"```$", "", s2).strip()
        try:
            return json.loads(s2)
        except Exception:
            return {}

def _validate_decision(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sid = str(d.get("id", "")).strip()
    keep = bool(d.get("keep", False))
    label = _normalize_label(d.get("label"))
    subtype = d.get("subtype") or None
    reason = str(d.get("reason") or "").strip() or None
    conf = d.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = None
    if conf is not None:
        conf = min(max(conf, 0.0), 1.0)
    return {"id": sid, "keep": keep, "label": label, "subtype": subtype, "reason": reason, "confidence": conf}

def _validate_addition(a: Dict[str, Any],
                       windows_by_id: Dict[str, Dict[str, Any]],
                       text_len: int) -> Optional[Dict[str, Any]]:
    win_id = str(a.get("win_id", "")).strip()
    if win_id not in windows_by_id:
        return None
    try:
        sr = int(a.get("start_rel"))
        er = int(a.get("end_rel"))
    except Exception:
        return None
    if not (0 <= sr < er <= len(windows_by_id[win_id]["context"])):
        return None

    # 전역 좌표 변환
    offset = int(windows_by_id[win_id]["offset"])
    s, e = offset + sr, offset + er
    if not (0 <= s < e <= text_len):
        return None
    if (e - s) > MAX_ADDITION_LEN:
        return None

    label = _normalize_label(a.get("label"))
    if not label:
        return None
    subtype = a.get("subtype") or None
    reason = str(a.get("reason") or "").strip() or None
    conf = a.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = None
    if conf is not None:
        conf = min(max(conf, 0.0), 1.0)

    return {"start": s, "end": e, "label": label, "label_adjusted": label,
            "subtype": subtype, "source": "gpt-add", "gpt_reason": reason, "gpt_confidence": conf}

# ────────────────────────────────────────────────────────────────────
# 겹침/중복 해소
# ────────────────────────────────────────────────────────────────────
def _priority(lbl: Optional[str]) -> int:
    return LABEL_PRIORITY.get(lbl or "", 0)

def _dedup_and_resolve(spans: List[Dict[str, Any]], text_len: int) -> List[Dict[str, Any]]:
    # 1) 범위 정규화/클리핑
    clean: List[Dict[str, Any]] = []
    for s in spans:
        try:
            a, b = int(s["start"]), int(s["end"])
        except Exception:
            continue
        if not (0 <= a < b <= text_len):
            continue
        lbl = _normalize_label(s.get("label_adjusted") or s.get("label"))
        if not lbl:
            continue
        ss = dict(s)
        ss["start"], ss["end"] = a, b
        ss["label_adjusted"] = lbl
        clean.append(ss)

    # 2) 같은 범위 중복 → 우선순위 높은 라벨 선택
    by_range: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for s in clean:
        key = (s["start"], s["end"])
        prev = by_range.get(key)
        if prev is None or _priority(s.get("label_adjusted")) > _priority(prev.get("label_adjusted")):
            by_range[key] = s
    merged = list(by_range.values())

    # 3) 부분 겹침 → 길이*우선순위 점수로 선택
    merged.sort(key=lambda x: (x["start"], x["end"]))
    out: List[Dict[str, Any]] = []
    for cur in merged:
        drop = False
        for j, kept in enumerate(out):
            if not (cur["end"] <= kept["start"] or cur["start"] >= kept["end"]):
                # 겹침 있음 → 점수 비교
                len_cur = cur["end"] - cur["start"]
                len_kept = kept["end"] - kept["start"]
                score_cur = _priority(cur["label_adjusted"]) * math.log1p(len_cur)
                score_kept = _priority(kept["label_adjusted"]) * math.log1p(len_kept)
                if score_cur > score_kept:
                    out[j] = cur
                drop = True
                break
        if not drop:
            out.append(cur)
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out

# ────────────────────────────────────────────────────────────────────
# GPT 호출
# ────────────────────────────────────────────────────────────────────
def _call_gpt(messages: List[Dict[str, str]],
              model: str,
              temperature: float,
              max_retries: int = 2) -> Optional[Dict[str, Any]]:
    if _client is None:
        return None

    delay = 0.2
    last_exc = None
    for _ in range(max_retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=model,
                temperature=float(temperature),
                top_p=1,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return _safe_json_load(resp.choices[0].message.content)
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay = min(delay * 2, 1.2)
    return None

# ────────────────────────────────────────────────────────────────────
# 메인 엔트리
# ────────────────────────────────────────────────────────────────────
def validate_with_gpt(text: str,
                      spans: List[Dict[str, Any]],
                      model: Optional[str] = None,
                      temperature: float = 0.0,
                      use_windows_when_coverage_below: float = 0.04) -> List[Dict[str, Any]]:

    model = model or DEFAULT_MODEL
    text_len = len(text)

    # 커버리지 계산
    covered = sum(max(0, min(text_len, s.get("end", 0)) - max(0, s.get("start", 0))) for s in spans) if spans else 0
    coverage = covered / max(1, text_len)

    # 윈도우 사용 여부 결정 (기존 0.01 → 0.04로 상향)
    use_windows = coverage < use_windows_when_coverage_below

    items = _build_items_masked(text, spans) if spans else []
    windows = _build_windows_masked(text) if use_windows else []

    # messages 생성
    messages = [
        {"role": "system", "content": "You are a precise Korean PII validator that never reveals PII and returns strict JSON."},
        {"role": "user", "content": _prompt_payload(items, windows if use_windows else None)},
    ]

    obj = _call_gpt(messages, model=model, temperature=temperature)
    if obj is None:
        # GPT 실패: 원본 스팬 정규화/정렬만 수행
        out = _dedup_and_resolve(spans, text_len)
        return out

    # id → span 매핑
    id2span = {str(i): dict(s) for i, s in enumerate(spans)}

    # windows map (offset 포함)
    windows_by_id = {w["id"]: w for w in windows}

    out: List[Dict[str, Any]] = []

    # 1) decisions 처리
    for d in obj.get("decisions", []) or []:
        vd = _validate_decision(d)
        if not vd:
            continue
        sid = vd["id"]
        if sid in id2span and vd["keep"] is True:
            base = dict(id2span[sid])
            lbl = _normalize_label(vd.get("label")) or _normalize_label(base.get("label_adjusted") or base.get("label"))
            if not lbl:
                continue
            base["label_adjusted"] = lbl
            base["subtype"] = vd.get("subtype") or None
            if vd.get("reason"):
                base["gpt_reason"] = vd["reason"]
            if vd.get("confidence") is not None:
                base["gpt_confidence"] = vd["confidence"]
            base["source"] = (base.get("source") or "regex") + "+gpt"
            out.append(base)

    # 2) additions 처리 (윈도우 상대 좌표 → 전역)
    for a in obj.get("additions", []) or []:
        va = _validate_addition(a, windows_by_id, text_len)
        if va:
            # 텍스트 포함 (검증/디버그용)
            va["text"] = text[va["start"]:va["end"]]
            out.append(va)

    # 3) 아무것도 없으면 폴백: 원본
    if not out:
        out = list(spans)

    # 4) 겹침/중복 해소 + 정렬
    out = _dedup_and_resolve(out, text_len)
    return out
