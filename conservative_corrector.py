# conservative_corrector.py
"""
보수적 STT 텍스트 보정 시스템
확실한 것만 수정하고, 애매한 부분은 사용자 검토 요청
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class SuspiciousPart:
    """의심스러운 부분"""
    text: str
    start: int
    end: int
    confidence: float
    reason: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    """보정 결과"""
    original_text: str
    corrected_text: str
    suspicious_parts: List[SuspiciousPart]
    auto_corrections: List[Dict]
    needs_review: bool


class ConservativeCorrector:
    """보수적 텍스트 보정기"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def create_conservative_prompt(self, text: str) -> str:
        """보수적 보정 프롬프트"""
        
        prompt = f"""STT 텍스트를 보수적으로 보정하세요.

원칙:
1. 확실한 것만 수정:
   - 문장부호 추가 (? . , !)
   - 명백한 띄어쓰기 오류
   - 중복된 조사 (을를 → 를)

2. 수정하지 말 것:
   - 애매한 단어 (발음 혼동 가능성이 있어도)
   - 문법적 어색함 (의미는 통하면)
   - 구어체 표현

3. 의심스러운 부분 표시:
   - 문맥상 이상하지만 확실하지 않은 부분
   - 발음 혼동 가능성이 있는 부분
   - 가능한 대안 제시

원본 텍스트:
{text}

JSON 응답:
{{
    "corrected_text": "보수적으로 보정된 텍스트",
    "auto_corrections": [
        {{"type": "punctuation", "original": "있어", "corrected": "있어?", "confidence": 0.95}}
    ],
    "suspicious_parts": [
        {{
            "text": "의심스러운 부분",
            "start_char": 시작위치,
            "end_char": 끝위치,
            "confidence": 0.3,
            "reason": "문맥상 부자연스러움",
            "suggestions": ["대안1", "대안2"]
        }}
    ]
}}"""
        
        return prompt
    
    def apply_basic_rules(self, text: str) -> Tuple[str, List[Dict]]:
        """기본 규칙 적용 (GPT 호출 전)"""
        
        corrections = []
        result = text
        
        # 1. 중복 조사 제거
        duplicates = [
            (r'을를', '를'),
            (r'이가', '가'),
            (r'은는', '는'),
        ]
        
        for pattern, replacement in duplicates:
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result)
                corrections.append({
                    "type": "duplicate",
                    "original": pattern,
                    "corrected": replacement,
                    "confidence": 1.0
                })
        
        # 2. 띄어쓰기 (매우 명백한 것만)
        # 예: "못 움직이게하려고" → "못 움직이게 하려고"
        
        return result, corrections
    
    def correct(self, text: str) -> CorrectionResult:
        """보수적 보정 실행"""
        
        # 기본 규칙 적용
        pre_corrected, rule_corrections = self.apply_basic_rules(text)
        
        # GPT 호출
        prompt = self.create_conservative_prompt(pre_corrected)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a conservative Korean text corrector. Only fix what you're certain about."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 매우 보수적
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 의심스러운 부분 파싱
            suspicious_parts = []
            for susp in result.get("suspicious_parts", []):
                suspicious_parts.append(SuspiciousPart(
                    text=susp["text"],
                    start=susp.get("start_char", 0),
                    end=susp.get("end_char", 0),
                    confidence=susp.get("confidence", 0.5),
                    reason=susp.get("reason", ""),
                    suggestions=susp.get("suggestions", [])
                ))
            
            # 자동 수정 사항 병합
            all_corrections = rule_corrections + result.get("auto_corrections", [])
            
            return CorrectionResult(
                original_text=text,
                corrected_text=result.get("corrected_text", pre_corrected),
                suspicious_parts=suspicious_parts,
                auto_corrections=all_corrections,
                needs_review=len(suspicious_parts) > 0
            )
            
        except Exception as e:
            print(f"❌ GPT 오류: {e}")
            # 실패 시 기본 규칙만 적용
            return CorrectionResult(
                original_text=text,
                corrected_text=pre_corrected,
                suspicious_parts=[],
                auto_corrections=rule_corrections,
                needs_review=False
            )


class InteractiveReviewer:
    """사용자 검토 인터페이스"""
    
    def __init__(self):
        self.review_history = []
    
    def display_for_review(self, result: CorrectionResult) -> Dict:
        """검토용 표시 데이터 생성"""
        
        display_data = {
            "text": result.corrected_text,
            "highlights": [],
            "review_needed": result.needs_review
        }
        
        # 의심스러운 부분 하이라이트 정보
        for susp in result.suspicious_parts:
            display_data["highlights"].append({
                "start": susp.start,
                "end": susp.end,
                "text": susp.text,
                "reason": susp.reason,
                "suggestions": susp.suggestions,
                "confidence": susp.confidence
            })
        
        # 자동 수정 사항 요약
        display_data["auto_corrections_summary"] = self._summarize_corrections(
            result.auto_corrections
        )
        
        return display_data
    
    def _summarize_corrections(self, corrections: List[Dict]) -> str:
        """자동 수정 사항 요약"""
        
        if not corrections:
            return "자동 수정 없음"
        
        summary = []
        for corr in corrections:
            if corr["type"] == "punctuation":
                summary.append("문장부호 추가")
            elif corr["type"] == "duplicate":
                summary.append("중복 조사 제거")
            elif corr["type"] == "spacing":
                summary.append("띄어쓰기 수정")
        
        return ", ".join(set(summary))
    
    def apply_user_corrections(self, result: CorrectionResult, 
                              user_choices: Dict[str, str]) -> str:
        """사용자 선택 적용"""
        
        final_text = result.corrected_text
        
        # 사용자 선택 적용 (역순으로 처리하여 인덱스 꼬임 방지)
        for original, replacement in sorted(
            user_choices.items(), 
            key=lambda x: final_text.rfind(x[0]), 
            reverse=True
        ):
            final_text = final_text.replace(original, replacement)
        
        # 기록 저장
        self.review_history.append({
            "original": result.original_text,
            "auto_corrected": result.corrected_text,
            "user_choices": user_choices,
            "final": final_text
        })
        
        return final_text


# 편의 함수
def conservative_correct_with_review(text: str, api_key: Optional[str] = None) -> Dict:
    """
    보수적 보정 + 검토 필요 여부 반환
    
    Returns:
        {
            "corrected": 보정된 텍스트,
            "needs_review": 검토 필요 여부,
            "review_items": 검토 항목들,
            "summary": 요약
        }
    """
    
    corrector = ConservativeCorrector(api_key)
    reviewer = InteractiveReviewer()
    
    # 보정 실행
    result = corrector.correct(text)
    
    # 검토용 데이터 생성
    display_data = reviewer.display_for_review(result)
    
    return {
        "corrected": result.corrected_text,
        "needs_review": result.needs_review,
        "review_items": display_data["highlights"],
        "auto_corrections": display_data["auto_corrections_summary"],
        "original": text
    }


# 테스트
if __name__ == "__main__":
    test_text = """최근에 건강은 어때 건강한 것 같아요. 최근에 더 친 곳 있어? 
    아빠가 꽉 잡아서 어깨가 아파요. 언제 다쳤어 이번 주 월요일이야"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ API 키를 설정하세요")
    else:
        print("="*60)
        print("🔍 보수적 보정 + 사용자 검토 시스템")
        print("="*60)
        
        # 보정 실행
        result = conservative_correct_with_review(test_text, api_key)
        
        print(f"\n📝 원본:")
        print(result["original"])
        
        print(f"\n✏️ 자동 보정:")
        print(result["corrected"])
        print(f"자동 수정: {result['auto_corrections']}")
        
        if result["needs_review"]:
            print(f"\n⚠️ 검토 필요: {len(result['review_items'])}개 항목")
            for item in result["review_items"]:
                print(f"\n  [{item['text']}]")
                print(f"  이유: {item['reason']}")
                print(f"  제안: {item['suggestions']}")
                print(f"  신뢰도: {item['confidence']:.0%}")
        else:
            print("\n✅ 검토 필요 없음")