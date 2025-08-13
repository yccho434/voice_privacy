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
    linguistic_type: str = ""  # 음운적/의미적/문법적


@dataclass
class CorrectionResult:
    """보정 결과"""
    original_text: str
    corrected_text: str
    suspicious_parts: List[SuspiciousPart]
    auto_corrections: List[Dict]
    needs_review: bool
    context_analysis: Dict = field(default_factory=dict)  # 문맥 분석 결과


class ConservativeCorrector:
    """보수적 텍스트 보정기"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def create_system_prompt(self) -> str:
        """시스템 프롬프트 - 역할 정의"""
        return """You are a Korean STT post-processor with expertise in:
1. Korean phonology and common misrecognition patterns
2. Natural language understanding for context analysis
3. Conservative correction - only fix what's certain

Your task is to:
- Identify potential STT errors based on phonetic similarity
- Consider semantic coherence within the context
- Mark suspicious parts rather than auto-correcting uncertain cases

Key principles:
- STT often confuses words with similar pronunciation
- Single syllables can be incorrectly separated or merged
- Context is crucial for identifying errors
- When in doubt, mark as suspicious rather than correct"""
    
    def create_analysis_prompt(self, text: str) -> str:
        """1단계: 문맥 분석 프롬프트"""
        return f"""다음 STT 출력 텍스트를 분석하세요:

텍스트:
{text}

분석 항목:
1. 전체 대화의 주제/상황 파악
2. 각 문장의 의미적 완성도
3. 문장 간 의미 연결성
4. 어색하거나 부자연스러운 표현

JSON 형식으로 답하세요:
{{
    "topic": "대화 주제",
    "context_type": "일상대화|의료상담|심리상담|업무대화|기타",
    "sentence_analysis": [
        {{
            "sentence": "문장",
            "semantic_completeness": 0.0-1.0,
            "issues": ["발견된 문제점"]
        }}
    ],
    "overall_coherence": 0.0-1.0
}}"""
    
    def create_conservative_prompt(self, text: str, context: Optional[Dict] = None) -> str:
        """2단계: 보수적 보정 프롬프트"""
        
        context_info = ""
        if context:
            context_info = f"""
문맥 정보:
- 주제: {context.get('topic', '불명')}
- 유형: {context.get('context_type', '일반')}
- 전체 일관성: {context.get('overall_coherence', 0):.1f}
"""
        
        prompt = f"""STT 텍스트를 보수적으로 보정하세요.
{context_info}

보정 원칙:
1. 확실한 것만 수정:
   - 문장부호 추가 (? . , !)
   - 명백한 띄어쓰기 오류
   - 중복된 조사 (을를 → 를)

2. 의심스러운 부분 탐지 (핵심):
   음운적 오류 가능성:
   - 발음 유사 단어 (ㄷ/ㅌ, ㅂ/ㅍ, ㄱ/ㅋ 혼동)
   - 연음으로 인한 오인식
   - 음절 분리/결합 오류
   
   의미적 오류 가능성:
   - 문맥과 맞지 않는 단어
   - 주어-서술어 의미 불일치
   - 전후 문장과 연결 안 되는 내용
   
   문법적 오류 가능성:
   - 조사 불일치 (받침 유무)
   - 어미 활용 오류
   - 어순 이상

3. 수정하지 말 것:
   - 구어체 표현
   - 방언이나 줄임말
   - 감정 표현

원본 텍스트:
{text}

분석 방법:
1. 각 문장을 음성으로 발화했을 때를 상상
2. 유사한 발음으로 오인식될 수 있는 부분 찾기
3. 전후 문맥과 의미가 자연스럽게 연결되는지 확인

JSON 응답:
{{
    "corrected_text": "보수적으로 보정된 텍스트",
    "auto_corrections": [
        {{
            "type": "punctuation|spacing|duplicate",
            "original": "원본 텍스트",
            "corrected": "수정된 텍스트",
            "confidence": 0.9,
            "position": 시작위치
        }}
    ],
    "suspicious_parts": [
        {{
            "text": "의심스러운 부분",
            "start_char": 시작위치,
            "end_char": 끝위치,
            "confidence": 0.3,
            "reason": "구체적 이유",
            "linguistic_type": "phonetic|semantic|grammatical",
            "suggestions": ["대안1", "대안2"],
            "context_clue": "판단 근거가 된 주변 문맥"
        }}
    ]
}}

중요: 
- confidence가 0.5 이하인 모든 의심 구간을 suspicious_parts에 포함
- 음운적으로 유사한 대안이 있다면 반드시 suggestions에 포함
- reason은 구체적이고 언어학적 근거를 제시"""
        
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
            (r'과와', '와'),
        ]
        
        for pattern, replacement in duplicates:
            if re.search(pattern, result):
                # 위치 찾기
                for match in re.finditer(pattern, result):
                    corrections.append({
                        "type": "duplicate",
                        "original": pattern,
                        "corrected": replacement,
                        "confidence": 1.0,
                        "position": match.start()
                    })
                result = re.sub(pattern, replacement, result)
        
        # 2. 명백한 띄어쓰기 (매우 보수적)
        # 예: "그런데요" → "그런데 요" (X, 구어체 보존)
        # 예: "했습니다.그래서" → "했습니다. 그래서" (O)
        result = re.sub(r'([.!?])([가-힣])', r'\1 \2', result)
        
        # 3. 연속 공백 제거
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result, corrections
    
    def analyze_context(self, text: str) -> Dict:
        """문맥 분석 (선택적)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Korean language expert analyzing STT output."
                    },
                    {"role": "user", "content": self.create_analysis_prompt(text)}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ 문맥 분석 실패: {e}")
            return {}
    
    def correct(self, text: str, use_context_analysis: bool = True) -> CorrectionResult:
        """보수적 보정 실행"""
        
        # 기본 규칙 적용
        pre_corrected, rule_corrections = self.apply_basic_rules(text)
        
        # 문맥 분석 (선택적)
        context = {}
        if use_context_analysis:
            print("🔍 문맥 분석 중...")
            context = self.analyze_context(pre_corrected)
        
        # GPT 보정 호출
        prompt = self.create_conservative_prompt(pre_corrected, context)
        
        try:
            print("✏️ 보정 분석 중...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.create_system_prompt()
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # 0.1에서 0.2로 약간 상향
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
                    suggestions=susp.get("suggestions", []),
                    linguistic_type=susp.get("linguistic_type", "")
                ))
            
            # 자동 수정 사항 병합
            all_corrections = rule_corrections + result.get("auto_corrections", [])
            
            return CorrectionResult(
                original_text=text,
                corrected_text=result.get("corrected_text", pre_corrected),
                suspicious_parts=suspicious_parts,
                auto_corrections=all_corrections,
                needs_review=len(suspicious_parts) > 0,
                context_analysis=context
            )
            
        except Exception as e:
            print(f"❌ GPT 오류: {e}")
            # 실패 시 기본 규칙만 적용
            return CorrectionResult(
                original_text=text,
                corrected_text=pre_corrected,
                suspicious_parts=[],
                auto_corrections=rule_corrections,
                needs_review=False,
                context_analysis=context
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
            "review_needed": result.needs_review,
            "context": result.context_analysis
        }
        
        # 의심스러운 부분 하이라이트 정보
        for susp in result.suspicious_parts:
            highlight = {
                "start": susp.start,
                "end": susp.end,
                "text": susp.text,
                "reason": susp.reason,
                "linguistic_type": susp.linguistic_type,
                "suggestions": susp.suggestions,
                "confidence": susp.confidence
            }
            
            # 언어학적 타입별 색상
            if susp.linguistic_type == "phonetic":
                highlight["color"] = "#FF6B6B"  # 빨강 - 음운적
            elif susp.linguistic_type == "semantic":
                highlight["color"] = "#4ECDC4"  # 청록 - 의미적
            elif susp.linguistic_type == "grammatical":
                highlight["color"] = "#45B7D1"  # 파랑 - 문법적
            else:
                highlight["color"] = "#FFA07A"  # 주황 - 기타
            
            display_data["highlights"].append(highlight)
        
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
        types = {}
        for corr in corrections:
            corr_type = corr.get("type", "기타")
            types[corr_type] = types.get(corr_type, 0) + 1
        
        for corr_type, count in types.items():
            if corr_type == "punctuation":
                summary.append(f"문장부호 {count}개")
            elif corr_type == "duplicate":
                summary.append(f"중복 조사 {count}개")
            elif corr_type == "spacing":
                summary.append(f"띄어쓰기 {count}개")
            else:
                summary.append(f"{corr_type} {count}개")
        
        return ", ".join(summary)
    
    def apply_user_corrections(self, result: CorrectionResult, 
                              user_choices: Dict[str, str]) -> str:
        """사용자 선택 적용"""
        
        final_text = result.corrected_text
        
        # 사용자 선택 적용 (역순으로 처리하여 인덱스 꼬임 방지)
        replacements = []
        for original, replacement in user_choices.items():
            # 텍스트에서 해당 부분 찾기
            idx = final_text.find(original)
            if idx >= 0:
                replacements.append((idx, original, replacement))
        
        # 위치 역순 정렬
        replacements.sort(reverse=True)
        
        # 치환 적용
        for idx, original, replacement in replacements:
            final_text = final_text[:idx] + replacement + final_text[idx + len(original):]
        
        # 기록 저장
        self.review_history.append({
            "original": result.original_text,
            "auto_corrected": result.corrected_text,
            "user_choices": user_choices,
            "final": final_text,
            "context": result.context_analysis
        })
        
        return final_text


# 편의 함수
def conservative_correct_with_review(text: str, api_key: Optional[str] = None,
                                    use_context: bool = True) -> Dict:
    """
    보수적 보정 + 검토 필요 여부 반환
    
    Args:
        text: STT 출력 텍스트
        api_key: OpenAI API 키
        use_context: 문맥 분석 사용 여부
    
    Returns:
        {
            "corrected": 보정된 텍스트,
            "needs_review": 검토 필요 여부,
            "review_items": 검토 항목들,
            "summary": 요약,
            "context": 문맥 분석 결과
        }
    """
    
    corrector = ConservativeCorrector(api_key)
    reviewer = InteractiveReviewer()
    
    # 보정 실행
    result = corrector.correct(text, use_context_analysis=use_context)
    
    # 검토용 데이터 생성
    display_data = reviewer.display_for_review(result)
    
    return {
        "corrected": result.corrected_text,
        "needs_review": result.needs_review,
        "review_items": display_data["highlights"],
        "auto_corrections": display_data["auto_corrections_summary"],
        "original": text,
        "context": display_data.get("context", {})
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
        
        if result.get("context"):
            print(f"\n📊 문맥 분석:")
            print(f"주제: {result['context'].get('topic', '불명')}")
            print(f"유형: {result['context'].get('context_type', '일반')}")
            print(f"일관성: {result['context'].get('overall_coherence', 0):.1f}")
        
        if result["needs_review"]:
            print(f"\n⚠️ 검토 필요: {len(result['review_items'])}개 항목")
            for i, item in enumerate(result["review_items"], 1):
                print(f"\n  {i}. [{item['text']}]")
                print(f"     유형: {item.get('linguistic_type', '일반')}")
                print(f"     이유: {item['reason']}")
                print(f"     제안: {item['suggestions']}")
                print(f"     신뢰도: {item['confidence']:.0%}")
        else:
            print("\n✅ 검토 필요 없음")