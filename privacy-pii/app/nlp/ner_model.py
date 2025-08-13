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

MODEL_ID = os.getenv("HF_NER_MODEL", "KPF/KPF-bert-ner")
CACHE_DIR = os.getenv("HF_CACHE_DIR", ".hf_cache")
HF_TOKEN = os.getenv("HF_TOKEN", None)

_DEVICE = 0 if torch.cuda.is_available() else -1
_PIPE: Optional[Any] = None

TAG2APP = {
    "PS_NAME": "이름",
    "OG_OTHERS": "소속", "OGG_ECONOMY": "소속","OGG_EDUCATION": "소속","OGG_MILITARY": "소속",
    "OGG_MEDIA": "소속","OGG_SPORTS": "소속","OGG_ART": "소속","OGG_MEDICINE": "소속",
    "OGG_RELIGION": "소속","OGG_SCIENCE": "소속","OGG_LIBRARY": "소속","OGG_LAW": "소속",
    "OGG_POLITICS": "소속","OGG_FOOD": "소속","OGG_HOTEL": "소속",
    "LC_OTHERS": "주소","LCP_COUNTRY": "주소","LCP_PROVINCE": "주소","LCP_COUNTY": "주소",
    "LCP_CITY": "주소","LCP_CAPITALCITY": "주소","LCG_RIVER": "주소","LCG_OCEAN": "주소",
    "LCG_BAY": "주소","LCG_MOUNTAIN": "주소","LCG_ISLAND": "주소","LCG_CONTINENT": "주소",
    "LC_TOUR": "주소","LC_SPACE": "주소",
    "QT_PHONE": "번호","QT_ZIPCODE": "번호",
    "TMI_EMAIL": "계정","TMI_SITE": "URL",
}

def _normalize_label(raw_label: str) -> str:
    lab = raw_label
    if "-" in lab:
        lab = lab.split("-", 1)[-1]
    return TAG2APP.get(lab, None)

def _build_pipeline() -> Optional[Any]:
    global _PIPE
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, token=HF_TOKEN, cache_dir=CACHE_DIR, trust_remote_code=True
        )
        mdl = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, token=HF_TOKEN, cache_dir=CACHE_DIR, trust_remote_code=True
        )
        _PIPE = pipeline(
            "token-classification", model=mdl, tokenizer=tok,
            device=_DEVICE, aggregation_strategy="simple",
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
        pipe = _ensure_pipe()
        if pipe is None:
            return []
        results = pipe(text)
        out = []
        for r in results:
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
