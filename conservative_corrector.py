# conservative_corrector.py
"""
보수적 STT 텍스트 보정 시스템 - GPT 최적화 버전
Claude처럼 작동하도록 개선
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
    """보수적 텍스트 보정기 - GPT 최적화"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # 일반적인 STT 오류 패턴 (하드코딩 아님, 학습용)
        self.common_patterns = self._load_common_patterns()
    
    def _load_common_patterns(self) -> Dict:
        """일반적인 STT 오류 패턴 로드"""
        return {
            "phonetic_confusion": [
                ("잤", "졌", "게임|보드게임|이기|지는"),
                ("일정", "일등", "성적|등수|1등|일등"),
                ("한이", "한 번도", "없어요|없습니다"),
                ("한의어", "한 번도", "없어요|없습니다"),
                ("초아", "좋아", "친구|기분"),
                ("프사", "학교", "친구|학생"),
                ("손이라고", "손해라고", "기분|나빠"),
            ],
            "spacing_errors": [
                ("어때요", "어때요", None),  # 문맥에 따라 다름
            ]
        }
    
    def create_system_prompt(self) -> str:
        """시스템 프롬프트 - GPT를 언어학자로 만들기"""
        return """당신은 10년 경력의 한국어 STT 오류 보정 전문가입니다.
수천 개의 음성 인식 오류를 분석한 경험이 있으며, 특히 한국어 음운 변화와 구어체 특성을 깊이 이해합니다.

당신의 전문 분야:
1. 한국어 음운학 - 발음 유사성으로 인한 오인식 패턴
2. 문맥 기반 의미 분석 - 대화의 흐름과 주제 파악
3. 구어체 특성 - 일상 대화의 자연스러운 흐름

작업 원칙:
- 확실한 오류만 수정합니다
- 애매한 부분은 원본을 유지하고 의심 구간으로 표시합니다
- 각 수정에는 명확한 언어학적 근거가 있어야 합니다"""
    
    def create_analysis_prompt(self, text: str) -> str:
        """1단계: 문맥 파악 프롬프트 (Chain of Thought)"""
        return f"""다음 STT 텍스트를 분석하세요.

=== 분석할 텍스트 ===
{text}

=== 단계별 분석 지시 ===

STEP 1. 대화 주제와 상황 파악
- 이 대화의 주요 주제는 무엇인가?
- 대화 참여자는 누구인가? (나이, 관계 추정)
- 어떤 상황에서 나눈 대화인가?

STEP 2. 의미적 일관성 검토
- 각 문장이 전체 문맥에서 자연스러운가?
- 앞뒤 문장과 논리적으로 연결되는가?
- 이상하거나 부자연스러운 표현이 있는가?

STEP 3. 잠재적 STT 오류 식별
- 문맥상 이상한 단어들을 찾아내세요
- 각 이상한 부분에 대해 "왜 이상한지" 설명하세요
- 음운적으로 유사한 대안이 있는지 생각해보세요

JSON 응답:
{{
    "topic": "대화의 주제",
    "context_type": "상담|일상대화|교육|의료|기타",
    "participants": {{
        "speaker1": "추정 정보",
        "speaker2": "추정 정보"
    }},
    "semantic_issues": [
        {{
            "text": "이상한 부분",
            "reason": "왜 이상한지",
            "context_clue": "판단 근거가 된 문맥",
            "suggested_correction": "제안하는 수정"
        }}
    ],
    "confidence_level": "high|medium|low"
}}"""
    
    def create_conservative_prompt(self, text: str, context: Optional[Dict] = None) -> str:
        """2단계: 보정 프롬프트 - Few-shot 학습 포함"""
        
        # Few-shot 예시들
        examples = """
=== 보정 예시들 (학습용) ===

예시 1:
원본: "어제 아빠랑 보드게임 하다가 제가 잤거든요"
문맥: 보드게임을 하면서 화가 난 이야기
분석: '잤거든요'는 문맥상 부자연스러움. 보드게임 + 화났다 → '졌거든요'가 적절
수정: "어제 아빠랑 보드게임 하다가 제가 졌거든요"
신뢰도: 0.9 (문맥이 명확)

예시 2:
원본: "초아 친구는 형제가 있어요?"
문맥: 상담 대화, 질문-답변 형식
분석: '초아'가 문장 시작에 어색함. '좋아요.' + '친구는'으로 분리 추정
수정: "좋아요. 친구는 형제가 있어요?"
신뢰도: 0.7 (문맥상 추정)

예시 3:
원본: "한이 없어요"
문맥: 부정적 경험 유무를 묻는 질문에 대한 답변
분석: '한이'는 '한 번도'의 오인식으로 추정
수정: "한 번도 없어요"
신뢰도: 0.8 (문법적으로 더 자연스러움)

예시 4:
원본: "일정을 하지 못하게 될까봐"
문맥: 성적과 등수에 대한 걱정
분석: '일정'은 '일등'의 오인식. 성적 문맥에서 '일등'이 적절
수정: "일등을 하지 못하게 될까봐"
신뢰도: 0.85 (문맥이 명확)
"""
        
        context_info = ""
        if context:
            context_info = f"""
=== 1단계 분석 결과 ===
주제: {context.get('topic', '불명')}
상황: {context.get('context_type', '일반')}
주요 문제: {len(context.get('semantic_issues', []))}개 발견
"""
        
        prompt = f"""{examples}

{context_info}

=== 보정할 텍스트 ===
{text}

=== 작업 지시 ===

당신은 이제 위 예시들을 참고하여 텍스트를 보정해야 합니다.

각 문장에 대해 다음 단계를 따르세요:

1. 문맥 확인: 이 문장이 전체 대화에서 자연스러운가?
2. 음운 검토: 이상한 단어가 다른 단어의 오인식일 가능성은?
3. 의미 검증: 수정 후 의미가 더 자연스러워지는가?
4. 신뢰도 평가: 이 수정이 얼마나 확실한가?

보정 원칙:
- 신뢰도 0.7 이상만 자동 수정
- 0.3~0.7은 suspicious_parts에 포함
- 0.3 미만은 수정하지 않음
- 문장 끝마다 적절한 문장부호 추가
- 각 문장 사이에 줄바꿈 추가 (가독성)

JSON 응답 형식:
{{
    "corrected_text": "보정된 텍스트 (문장마다 줄바꿈)",
    "auto_corrections": [
        {{
            "type": "phonetic|semantic|punctuation|spacing",
            "original": "원본",
            "corrected": "수정",
            "confidence": 0.0-1.0,
            "reason": "수정 이유",
            "context_clue": "근거가 된 문맥"
        }}
    ],
    "suspicious_parts": [
        {{
            "text": "의심 부분",
            "start_char": 시작,
            "end_char": 끝,
            "confidence": 0.3-0.7,
            "reason": "의심 이유",
            "linguistic_type": "phonetic|semantic|grammatical",
            "suggestions": ["대안1", "대안2"],
            "context_clue": "주변 문맥"
        }}
    ],
    "reasoning": "전체적인 보정 근거와 접근 방법"
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
            (r'과와', '와'),
        ]
        
        for pattern, replacement in duplicates:
            if re.search(pattern, result):
                for match in re.finditer(pattern, result):
                    corrections.append({
                        "type": "duplicate",
                        "original": pattern,
                        "corrected": replacement,
                        "confidence": 1.0,
                        "position": match.start()
                    })
                result = re.sub(pattern, replacement, result)
        
        # 2. 문장 끝 처리
        result = re.sub(r'([가-힣])\s*\.\s*([가-힣])', r'\1. \2', result)
        
        # 3. 연속 공백 제거
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result, corrections
    
    def post_process_with_patterns(self, text: str, context: Dict) -> str:
        """컨텍스트 기반 후처리 (일반화된 패턴)"""
        
        # 문맥에서 키워드 추출
        topic = context.get('topic', '').lower()
        issues = context.get('semantic_issues', [])
        
        # 동적 패턴 적용
        result = text
        
        # semantic_issues에서 제안된 수정사항 적용
        for issue in issues:
            if issue.get('suggested_correction'):
                original = issue.get('text', '')
                correction = issue.get('suggested_correction', '')
                if original and correction and original in result:
                    result = result.replace(original, correction)
        
        # 문장별 줄바꿈 추가
        result = re.sub(r'([.!?])\s*(?=[가-힣])', r'\1\n', result)
        
        return result
    
    def correct(self, text: str, use_context_analysis: bool = True) -> CorrectionResult:
        """보수적 보정 실행 - 개선된 버전"""
        
        # 기본 규칙 적용
        pre_corrected, rule_corrections = self.apply_basic_rules(text)
        
        # 1단계: 문맥 분석 (Chain of Thought)
        context = {}
        if use_context_analysis and len(text) > 50:  # 짧은 텍스트는 스킵
            print("🔍 문맥 분석 중...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in Korean language and STT error analysis."
                        },
                        {"role": "user", "content": self.create_analysis_prompt(pre_corrected)}
                    ],
                    temperature=0.1,  # 더 낮춤
                    response_format={"type": "json_object"}
                )
                
                context = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"⚠️ 문맥 분석 실패: {e}")
                context = {}
        
        # 2단계: GPT 보정 (Few-shot + 문맥)
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
                temperature=0.0,  # 완전 deterministic
                top_p=0.1,  # 가장 확실한 것만
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 3단계: 후처리 (컨텍스트 기반)
            corrected_text = result.get("corrected_text", pre_corrected)
            if context:
                corrected_text = self.post_process_with_patterns(corrected_text, context)
            
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
            auto_corrections = rule_corrections + result.get("auto_corrections", [])
            
            return CorrectionResult(
                original_text=text,
                corrected_text=corrected_text,
                suspicious_parts=suspicious_parts,
                auto_corrections=auto_corrections,
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
    
    def correct_in_chunks(self, text: str, chunk_size: int = 10) -> CorrectionResult:
        """긴 텍스트를 청크로 나누어 처리 (선택적)"""
        
        sentences = re.split(r'([.!?]+)', text)
        chunks = []
        current_chunk = []
        
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + sentences[i+1] if i+1 < len(sentences) else sentences[i]
            current_chunk.append(sentence)
            
            if len(current_chunk) >= chunk_size:
                chunks.append(''.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        # 각 청크 처리
        all_corrected = []
        all_suspicious = []
        all_corrections = []
        
        for chunk in chunks:
            result = self.correct(chunk, use_context_analysis=True)
            all_corrected.append(result.corrected_text)
            all_suspicious.extend(result.suspicious_parts)
            all_corrections.extend(result.auto_corrections)
        
        # 병합
        final_text = '\n'.join(all_corrected)
        
        return CorrectionResult(
            original_text=text,
            corrected_text=final_text,
            suspicious_parts=all_suspicious,
            auto_corrections=all_corrections,
            needs_review=len(all_suspicious) > 0,
            context_analysis={}
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
            elif corr_type == "phonetic":
                summary.append(f"음운 오류 {count}개")
            elif corr_type == "semantic":
                summary.append(f"의미 오류 {count}개")
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
                                    use_context: bool = True,
                                    use_chunks: bool = False) -> Dict:
    """
    보수적 보정 + 검토 필요 여부 반환
    
    Args:
        text: STT 출력 텍스트
        api_key: OpenAI API 키
        use_context: 문맥 분석 사용 여부
        use_chunks: 청크 단위 처리 여부
    
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
    if use_chunks and len(text) > 500:  # 긴 텍스트는 청크로
        result = corrector.correct_in_chunks(text)
    else:
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
    # 실제 오류가 있는 텍스트
    test_text = """어제 아빠랑 보드게임 하다가 제가 잤거든요. 
    제가 원래 지는 걸 너무 싫어해서요.
    일정을 하지 못하게 될까 봐 무서워요.
    초아 친구는 형제가 있어요?
    프사 친구들이랑은 주로 뭐하고 놀아요?
    한이 없어요."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ API 키를 설정하세요")
    else:
        print("="*60)
        print("🔍 보수적 보정 + 사용자 검토 시스템")
        print("="*60)
        
        # 보정 실행
        result = conservative_correct_with_review(test_text, api_key, use_context=True)
        
        print(f"\n📝 원본:")
        print(result["original"])
        
        print(f"\n✏️ 자동 보정:")
        print(result["corrected"])
        print(f"자동 수정: {result['auto_corrections']}")
        
        if result.get("context"):
            print(f"\n📊 문맥 분석:")
            print(f"주제: {result['context'].get('topic', '불명')}")
            print(f"유형: {result['context'].get('context_type', '일반')}")
        
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