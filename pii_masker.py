# pii_masker.py
"""
PII 마스킹 처리 모듈
사용자가 UI에서 실시간으로 마스킹 규칙을 정의
"""

import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class MaskingRecord:
    """마스킹 기록"""
    entity_id: str
    original_text: str
    masked_text: str
    label: str
    subtype: Optional[str]
    start: int
    end: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class MaskingResult:
    """마스킹 결과"""
    original_text: str
    masked_text: str
    masking_records: List[MaskingRecord]
    mapping_file: Optional[str]
    timestamp: str
    stats: Dict = field(default_factory=dict)


class PIIMasker:
    """PII 마스킹 처리기 - 완전 유연한 구조"""
    
    def __init__(self, 
                 save_mapping: bool = True,
                 mapping_dir: str = "./masking_records"):
        """
        Args:
            save_mapping: 원본 매핑 저장 여부
            mapping_dir: 매핑 파일 저장 경로
        """
        self.save_mapping = save_mapping
        self.mapping_dir = Path(mapping_dir)
        
        if save_mapping:
            self.mapping_dir.mkdir(parents=True, exist_ok=True)
    
    def mask(self,
             text: str,
             entities: List[Dict],
             masking_rules: Dict[str, Any]) -> MaskingResult:
        """
        PII 마스킹 실행
        
        Args:
            text: 원본 텍스트
            entities: 4단계에서 탐지된 엔티티
            masking_rules: 사용자 정의 마스킹 규칙
                {
                    "mode": "simple|advanced|custom",
                    "simple_rule": "모든 PII를 이 텍스트로",  # simple mode
                    "label_rules": {  # advanced mode
                        "이름": "치환할 텍스트",
                        "번호": "치환할 텍스트",
                        ...
                    },
                    "entity_rules": {  # custom mode
                        "김철수": "홍길동",
                        "010-1234-5678": "전화번호",
                        ...
                    },
                    "default": "기본 마스킹 텍스트"
                }
        
        Returns:
            MaskingResult
        """
        # 타임스탬프
        timestamp = datetime.now().isoformat()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 엔티티 정렬 (뒤에서부터 치환)
        sorted_entities = sorted(entities, key=lambda x: x.get("end", 0), reverse=True)
        
        # 마스킹 처리
        masked_text = text
        masking_records = []
        stats = {"total": 0, "by_label": {}}
        
        for i, entity in enumerate(sorted_entities):
            # 엔티티 정보 추출
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            original = text[start:end]
            label = entity.get("label_adjusted") or entity.get("label", "기타")
            
            # 마스킹 텍스트 결정
            masked = self._determine_mask(original, entity, masking_rules)
            
            # 텍스트 치환
            masked_text = masked_text[:start] + masked + masked_text[end:]
            
            # 기록 생성
            entity_id = f"{session_id}_{i:04d}"
            record = MaskingRecord(
                entity_id=entity_id,
                original_text=original,
                masked_text=masked,
                label=label,
                subtype=entity.get("subtype"),
                start=start,
                end=end,
                metadata={
                    "source": entity.get("source", "unknown"),
                    "confidence": entity.get("score", 0.0)
                }
            )
            masking_records.append(record)
            
            # 통계 업데이트
            stats["total"] += 1
            stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
        
        # 매핑 저장
        mapping_file = None
        if self.save_mapping and masking_records:
            mapping_file = self._save_mapping(
                session_id, masking_records, text, masked_text, masking_rules
            )
        
        # 결과 생성
        return MaskingResult(
            original_text=text,
            masked_text=masked_text,
            masking_records=masking_records,
            mapping_file=mapping_file,
            timestamp=timestamp,
            stats=stats
        )
    
    def _determine_mask(self, 
                       original: str, 
                       entity: Dict, 
                       rules: Dict[str, Any]) -> str:
        """마스킹 텍스트 결정"""
        
        mode = rules.get("mode", "simple")
        
        if mode == "simple":
            # 모든 PII를 동일하게 마스킹
            return rules.get("simple_rule", "[MASKED]")
        
        elif mode == "advanced":
            # 라벨별 마스킹
            label = entity.get("label_adjusted") or entity.get("label", "기타")
            label_rules = rules.get("label_rules", {})
            
            if label in label_rules:
                return label_rules[label]
            else:
                return rules.get("default", "[MASKED]")
        
        elif mode == "custom":
            # 개별 엔티티 마스킹
            entity_rules = rules.get("entity_rules", {})
            
            if original in entity_rules:
                return entity_rules[original]
            else:
                # 라벨 규칙으로 폴백
                label = entity.get("label_adjusted") or entity.get("label", "기타")
                label_rules = rules.get("label_rules", {})
                
                if label in label_rules:
                    return label_rules[label]
                else:
                    return rules.get("default", "[MASKED]")
        
        else:
            return rules.get("default", "[MASKED]")
    
    def _save_mapping(self,
                     session_id: str,
                     records: List[MaskingRecord],
                     original: str,
                     masked: str,
                     rules: Dict) -> str:
        """마스킹 매핑 저장"""
        
        mapping_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "original_text": original,
            "masked_text": masked,
            "masking_rules": rules,  # 사용된 규칙도 저장
            "records": [asdict(r) for r in records],
            "stats": {
                "total_masked": len(records),
                "text_length": len(original),
                "masked_length": len(masked)
            }
        }
        
        # 파일 저장
        filename = self.mapping_dir / f"masking_{session_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        return str(filename)
    
    def restore(self, mapping_file: str) -> Dict:
        """매핑 파일에서 원본 복원"""
        
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        
        # 원본 텍스트 직접 반환 (정확한 복원)
        return {
            "original": mapping_data["original_text"],
            "masked": mapping_data["masked_text"],
            "rules_used": mapping_data.get("masking_rules", {}),
            "stats": mapping_data.get("stats", {})
        }