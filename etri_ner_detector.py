# etri_ner_detector.py
"""
ETRI 개체명 인식 API를 중심으로 한 PII 탐지 모듈
ETRI NER이 메인, 정규식은 보조 역할
"""

import json
import urllib3
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class PIIEntity:
    """PII 엔티티 데이터 클래스"""
    text: str
    label: str
    subtype: str
    start: int
    end: int
    score: float
    source: str  # 'etri', 'regex', 'merged'
    metadata: Dict = None


class ETRILanguageAnalyzer:
    """ETRI 언어 분석 API 클라이언트 - 메인 탐지 엔진"""
    
    # API 엔드포인트
    API_URL_WRITTEN = "http://epretx.etri.re.kr:8000/api/WiseNLU"
    API_URL_SPOKEN = "http://epretx.etri.re.kr:8000/api/WiseNLU_spoken"
    
    # ETRI 개체명 태그 → PII 라벨 매핑 (확장)
    TAG_TO_PII = {
        # === 개인 신원 ===
        "PS_NAME": ("이름", "인명"),
        
        # === 연락처/계정 ===
        "QT_PHONE": ("번호", "전화번호"),
        "TMI_EMAIL": ("계정", "이메일"),
        "TMI_SITE": ("URL", "웹사이트"),
        
        # === 주소/위치 ===
        "QT_ZIPCODE": ("주소", "우편번호"),
        "LCP_COUNTRY": ("주소", "국가"),
        "LCP_PROVINCE": ("주소", "도/주"),
        "LCP_COUNTY": ("주소", "군/구/동"),
        "LCP_CITY": ("주소", "도시"),
        "LC_OTHERS": ("주소", "기타장소"),
        "AF_BUILDING": ("주소", "건물명"),
        
        # === 소속/조직 ===
        "OGG_ECONOMY": ("소속", "기업"),
        "OGG_EDUCATION": ("소속", "교육기관"),
        "OGG_POLITICS": ("소속", "정부기관"),
        "OGG_MEDICINE": ("소속", "의료기관"),
        "OGG_MEDIA": ("소속", "언론사"),
        "OG_OTHERS": ("소속", "기타기관"),
        
        # === 금융 ===
        "QT_PRICE": ("금융", "금액"),
        "CV_CURRENCY": ("금융", "통화"),
        
        # === 개인정보 ===
        "QT_AGE": ("신원", "나이"),
        "TMM_DISEASE": ("신원", "건강정보"),
        "DT_DAY": ("신원", "날짜"),
        "TI_HOUR": ("신원", "시간"),
        "CV_OCCUPATION": ("신원", "직업"),
        "CV_POSITION": ("신원", "직위"),
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http = urllib3.PoolManager()
    
    def analyze(self, text: str, use_spoken: bool = False) -> Dict:
        """
        ETRI 언어 분석 API 호출
        
        Args:
            text: 분석할 텍스트
            use_spoken: 구어체 모드 사용 여부
        
        Returns:
            API 응답 결과
        """
        url = self.API_URL_SPOKEN if use_spoken else self.API_URL_WRITTEN
        
        # NER 분석 요청
        request_json = {
            "argument": {
                "text": text,
                "analysis_code": "ner"  # 개체명 인식
            }
        }
        
        try:
            response = self.http.request(
                "POST",
                url,
                headers={
                    "Content-Type": "application/json; charset=UTF-8",
                    "Authorization": self.api_key
                },
                body=json.dumps(request_json)
            )
            
            if response.status == 200:
                result = json.loads(response.data.decode("utf-8"))
                if result.get("result", -1) == 0:
                    return result.get("return_object", {})
                else:
                    print(f"❌ API 오류: {result.get('reason', 'Unknown error')}")
                    return {}
            else:
                print(f"❌ HTTP 오류: {response.status}")
                return {}
                
        except Exception as e:
            print(f"❌ 요청 실패: {str(e)}")
            return {}
    
    def extract_entities(self, text: str, use_spoken: bool = False) -> List[PIIEntity]:
        """
        ETRI NER을 사용한 개체명 추출 (정확한 위치 계산 포함)
        
        Args:
            text: 분석할 텍스트
            use_spoken: 구어체 모드
        
        Returns:
            PIIEntity 리스트
        """
        result = self.analyze(text, use_spoken)
        entities = []
        
        if not result:
            return entities
        
        # 문장별 처리
        current_pos = 0  # 전체 텍스트에서의 현재 위치
        
        for sentence in result.get("sentence", []):
            sentence_text = sentence.get("text", "")
            sentence_start = text.find(sentence_text, current_pos)
            
            if sentence_start == -1:
                continue
            
            # 형태소 위치 정보 구축
            morp_positions = []
            morp_offset = 0
            
            for morp_item in sentence.get("morp", []):
                morp_text = morp_item.get("lemma", "")
                morp_start = sentence_text.find(morp_text, morp_offset)
                if morp_start != -1:
                    morp_positions.append({
                        "id": morp_item.get("id"),
                        "text": morp_text,
                        "start": sentence_start + morp_start,
                        "end": sentence_start + morp_start + len(morp_text)
                    })
                    morp_offset = morp_start + len(morp_text)
            
            # NE (Named Entity) 정보 추출
            for ne in sentence.get("NE", []):
                ne_type = ne.get("type", "")
                ne_text = ne.get("text", "")
                
                # PII 관련 태그만 처리
                if ne_type in self.TAG_TO_PII:
                    label, subtype = self.TAG_TO_PII[ne_type]
                    
                    # begin, end는 형태소 ID 참조
                    begin_id = ne.get("begin", 0)
                    end_id = ne.get("end", 0)
                    
                    # 형태소 ID로 실제 위치 찾기
                    ne_start = -1
                    ne_end = -1
                    
                    for morp_pos in morp_positions:
                        if morp_pos["id"] == begin_id:
                            ne_start = morp_pos["start"]
                        if morp_pos["id"] == end_id:
                            ne_end = morp_pos["end"]
                    
                    # 위치를 못 찾은 경우 텍스트 검색으로 폴백
                    if ne_start == -1 or ne_end == -1:
                        found_pos = text.find(ne_text, current_pos)
                        if found_pos != -1:
                            ne_start = found_pos
                            ne_end = found_pos + len(ne_text)
                    
                    if ne_start != -1 and ne_end != -1:
                        entities.append(PIIEntity(
                            text=ne_text,
                            label=label,
                            subtype=subtype,
                            start=ne_start,
                            end=ne_end,
                            score=0.95,  # ETRI NER은 높은 신뢰도
                            source="etri",
                            metadata={
                                "etri_tag": ne_type,
                                "sentence_id": sentence.get("id")
                            }
                        ))
            
            current_pos = sentence_start + len(sentence_text)
        
        return entities


class SupplementaryRegexDetector:
    """ETRI가 놓칠 수 있는 패턴을 보완하는 정규식 탐지기"""
    
    # 한국 특화 패턴들
    PATTERNS = [
        # 주민등록번호 (ETRI가 잘 못 잡는 경우가 있음)
        (r'\b\d{6}[-\s]?\d{7}\b', "신원", "주민등록번호"),
        
        # 계좌번호 (은행명과 함께)
        (r'(?:국민|신한|우리|하나|기업|농협|카카오뱅크|토스뱅크)\s*\d{2,6}[-\s]\d{2,6}[-\s]\d{2,6}', "금융", "계좌번호"),
        
        # 신용카드번호
        (r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b', "금융", "카드번호"),
        
        # 운전면허번호
        (r'\b\d{2}[-\s]\d{6}[-\s]\d{2}\b', "신원", "운전면허"),
        
        # 여권번호
        (r'\b[MS]\d{8}\b', "신원", "여권번호"),
        
        # 사업자등록번호
        (r'\b\d{3}[-\s]\d{2}[-\s]\d{5}\b', "신원", "사업자번호"),
        
        # IP 주소
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', "URL", "IP주소"),
    ]
    
    def detect(self, text: str, etri_entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        ETRI가 놓친 패턴 탐지 (중복 제거)
        
        Args:
            text: 원본 텍스트
            etri_entities: ETRI가 찾은 엔티티들
        
        Returns:
            추가 PIIEntity 리스트
        """
        additional_entities = []
        
        # ETRI가 찾은 영역 기록
        covered_ranges = set()
        for entity in etri_entities:
            for i in range(entity.start, entity.end):
                covered_ranges.add(i)
        
        # 패턴 매칭
        for pattern, label, subtype in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                # ETRI가 이미 찾은 영역인지 확인
                if any(i in covered_ranges for i in range(start, end)):
                    continue
                
                additional_entities.append(PIIEntity(
                    text=match.group(),
                    label=label,
                    subtype=subtype,
                    start=start,
                    end=end,
                    score=0.85,
                    source="regex",
                    metadata={"pattern": pattern}
                ))
        
        return additional_entities


class EnhancedPIIDetector:
    """ETRI 중심의 향상된 PII 탐지기"""
    
    def __init__(self, etri_api_key: str):
        """
        Args:
            etri_api_key: ETRI API 키
        """
        self.etri = ETRILanguageAnalyzer(etri_api_key)
        self.regex = SupplementaryRegexDetector()
    
    def detect(self, text: str, use_spoken: bool = False) -> List[PIIEntity]:
        """
        ETRI 중심 PII 탐지
        
        Args:
            text: 분석할 텍스트
            use_spoken: 구어체 모드
        
        Returns:
            통합된 PIIEntity 리스트
        """
        print("🤖 ETRI NER API 호출 중...")
        
        # 1. ETRI NER이 메인 (높은 정확도)
        etri_entities = self.etri.extract_entities(text, use_spoken)
        print(f"  → ETRI: {len(etri_entities)}개 탐지")
        
        # 2. 정규식으로 보완 (ETRI가 놓친 것만)
        print("🔍 보완 패턴 검사 중...")
        additional_entities = self.regex.detect(text, etri_entities)
        print(f"  → 추가: {len(additional_entities)}개 탐지")
        
        # 3. 통합
        all_entities = etri_entities + additional_entities
        
        # 4. 후처리: 겹치는 엔티티 정리
        merged_entities = self._merge_overlapping(all_entities)
        print(f"✅ 최종: {len(merged_entities)}개 PII 확정")
        
        return merged_entities
    
    def _merge_overlapping(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        겹치는 엔티티 병합 (ETRI 우선)
        """
        if not entities:
            return []
        
        # 정렬: 시작 위치, ETRI 우선, 긴 것 우선
        entities.sort(key=lambda e: (
            e.start,
            0 if e.source == "etri" else 1,
            -(e.end - e.start)
        ))
        
        merged = []
        for entity in entities:
            # 겹치는 기존 엔티티 확인
            overlap = False
            for existing in merged:
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    # 겹침 - ETRI가 우선
                    if entity.source == "etri" and existing.source != "etri":
                        merged.remove(existing)
                        merged.append(entity)
                    overlap = True
                    break
            
            if not overlap:
                merged.append(entity)
        
        # 최종 정렬
        merged.sort(key=lambda e: e.start)
        return merged
    
    def format_results(self, entities: List[PIIEntity]) -> Dict:
        """
        결과를 보기 좋게 포맷팅
        """
        result = defaultdict(list)
        
        for entity in entities:
            category = f"{entity.label}"
            if entity.subtype:
                category += f" ({entity.subtype})"
            
            result[category].append({
                "text": entity.text,
                "position": f"{entity.start}-{entity.end}",
                "source": entity.source,
                "confidence": f"{entity.score:.0%}"
            })
        
        # 통계 추가
        result["_statistics"] = {
            "total": len(entities),
            "by_source": defaultdict(int),
            "by_label": defaultdict(int)
        }
        
        for entity in entities:
            result["_statistics"]["by_source"][entity.source] += 1
            result["_statistics"]["by_label"][entity.label] += 1
        
        return dict(result)


# 테스트 코드
if __name__ == "__main__":
    # 테스트 텍스트
    test_text = """
    안녕하세요, 김민수입니다. 제 연락처는 010-1234-5678이고, 
    이메일은 minsu.kim@example.com입니다. 
    서울특별시 강남구 테헤란로 123에 있는 삼성전자에서 일하고 있습니다.
    계좌번호는 우리은행 1002-345-678901입니다.
    주민번호는 801010-1234567이고요.
    다음 미팅은 2025-08-15 오후 3시입니다.
    """
    
    # API 키 설정
    import os
    api_key = os.getenv("ETRI_API_KEY", "YOUR_API_KEY")
    
    if api_key == "YOUR_API_KEY":
        print("⚠️ ETRI API 키를 설정해주세요!")
        print("export ETRI_API_KEY='your_actual_key'")
    else:
        # 탐지기 생성 및 실행
        detector = EnhancedPIIDetector(api_key)
        entities = detector.detect(test_text, use_spoken=False)
        
        # 결과 출력
        print("\n" + "="*50)
        print("📊 PII 탐지 결과")
        print("="*50)
        
        formatted = detector.format_results(entities)
        
        # 통계 출력
        stats = formatted.pop("_statistics")
        print(f"\n📈 통계:")
        print(f"  • 전체: {stats['total']}개")
        print(f"  • ETRI: {stats['by_source'].get('etri', 0)}개")
        print(f"  • 정규식: {stats['by_source'].get('regex', 0)}개")
        
        # 카테고리별 출력
        for category, items in formatted.items():
            print(f"\n🏷️ {category}:")
            for item in items:
                print(f"  • '{item['text']}' (위치: {item['position']}, 출처: {item['source']}, 신뢰도: {item['confidence']})")