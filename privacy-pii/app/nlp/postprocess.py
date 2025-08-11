# nlp/postprocess.py
from typing import List, Dict

def merge_spans(spans: List[Dict]) -> List[Dict]:
    """
    시작 위치 기준 정렬 후, 겹치면 긴 span 우선으로 간단 머지
    """
    spans = sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    merged = []
    last_end = -1
    for s in spans:
        if s["start"] >= last_end:
            merged.append(s)
            last_end = s["end"]
    return merged
