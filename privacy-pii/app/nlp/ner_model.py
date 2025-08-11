# nlp/ner_model.py
import os
import logging
from typing import List, Dict, Any, Optional

import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

load_dotenv()
logger = logging.getLogger(__name__)

# 환경변수 고정
MODEL_ID = os.getenv("HF_NER_MODEL", "KPF/KPF-bert-ner")
CACHE_DIR = os.getenv("HF_CACHE_DIR", ".hf_cache")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# 장치 선택
_DEVICE = 0 if torch.cuda.is_available() else -1

# 전역 파이프라인 핸들
_PIPE: Optional[Any] = None

# KPF 계열 라벨 → 앱 라벨 맵핑 (필요시 추가/수정)
# KPF 태그 예: PS_NAME, LCP_* (국가/도시/구/동), OGG_* (기관), QT_PHONE, TMI_EMAIL, TMI_SITE 등
TAG2APP = {
    # 인명
    "PS_NAME": "이름",

    # 기관(소속)
    "OG_OTHERS": "소속",
    "OGG_ECONOMY": "소속",
    "OGG_EDUCATION": "소속",
    "OGG_MILITARY": "소속",
    "OGG_MEDIA": "소속",
    "OGG_SPORTS": "소속",
    "OGG_ART": "소속",
    "OGG_MEDICINE": "소속",
    "OGG_RELIGION": "소속",
    "OGG_SCIENCE": "소속",
    "OGG_LIBRARY": "소속",
    "OGG_LAW": "소속",
    "OGG_POLITICS": "소속",
    "OGG_FOOD": "소속",
    "OGG_HOTEL": "소속",

    # 위치(주소 계열을 넓게 묶음)
    "LC_OTHERS": "주소",
    "LCP_COUNTRY": "주소",
    "LCP_PROVINCE": "주소",
    "LCP_COUNTY": "주소",
    "LCP_CITY": "주소",
    "LCP_CAPITALCITY": "주소",
    "LCG_RIVER": "주소",
    "LCG_OCEAN": "주소",
    "LCG_BAY": "주소",
    "LCG_MOUNTAIN": "주소",
    "LCG_ISLAND": "주소",
    "LCG_CONTINENT": "주소",
    "LC_TOUR": "주소",
    "LC_SPACE": "주소",

    # 연락처/계정/URL
    "QT_PHONE": "번호",
    "QT_ZIPCODE": "번호",  # 우편번호도 번호로 묶음
    "TMI_EMAIL": "계정",
    "TMI_SITE": "URL",

    # 금융(계좌/은행명 등은 KPF 태그에 직접 없어서 NER로 잡히면 후처리에서 보완)
    # 필요 시 규칙/키워드로 라벨 보정
}

def _normalize_label(raw_label: str) -> str:
    """
    모델 라벨(예: B-PS_NAME/I-PS_NAME/PS_NAME)을 앱 라벨로 정규화.
    """
    # BIO 접두사 제거
    lab = raw_label
    if "-" in lab:
        # 예: B-PS_NAME -> PS_NAME
        lab = lab.split("-", 1)[-1]
    return TAG2APP.get(lab, None)  # 매핑 없으면 None (하이라이트 제외)

def _build_pipeline() -> Optional[Any]:
    global _PIPE
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        mdl = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        _PIPE = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            device=_DEVICE,
            aggregation_strategy="simple",  # 엔티티 span 통합
        )
        logger.info(f"NER pipeline ready: {MODEL_ID} (device={_DEVICE})")
    except Exception as e:
        logger.exception("Failed to load NER pipeline")
        _PIPE = None
    return _PIPE

def _ensure_pipe():
    return _PIPE or _build_pipeline()

class NER:
    @staticmethod
    def available() -> bool:
        return _ensure_pipe() is not None

    @staticmethod
    def infer(text: str) -> List[Dict[str, Any]]:
        """
        출력 포맷: [{"start":int, "end":int, "text":str, "label":str, "score":float}]
        여기서 label은 앱 라벨(이름/주소/소속/계정/번호/URL 등)로 맵핑됨.
        맵핑 실패(None)는 필터링해서 반환.
        """
        pipe = _ensure_pipe()
        if pipe is None:
            return []

        results = pipe(text)
        out = []
        for r in results:
            # transformers pipeline 결과 키: entity_group, score, word, start, end
            raw = r.get("entity_group") or r.get("entity") or ""
            app_label = _normalize_label(raw)
            if app_label is None:
                continue
            out.append({
                "start": int(r["start"]),
                "end": int(r["end"]),
                "text": text[int(r["start"]):int(r["end"])],
                "label": app_label,
                "score": float(r.get("score", 0.0)),
            })
        return out
