# app/detectors/name_heuristics.py
from typing import List, Dict, Any
from app.nlp.tokenization import HybridTokenizer, HybridTokens

def detect_names_via_morph(ht: HybridTokens) -> List[Dict[str, Any]]:
    spans = ht  # alias
    hits = []
    # HybridTokenizer의 find_korean_name_spans를 사용할 계획(현 구현 보류)
    return hits
# app/detectors/name_heuristics.py
from typing import List, Dict, Any
from app.nlp.tokenization import HybridTokenizer, HybridTokens

def detect_names_via_morph(ht: HybridTokens) -> List[Dict[str, Any]]:
    spans = ht  # alias
    hits = []
    # HybridTokenizer의 find_korean_name_spans를 사용할 계획(현 구현 보류)
    return hits
