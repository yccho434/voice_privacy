# integrated_pipeline.py
"""
음성 프라이버시 보호 통합 파이프라인
1단계(음성향상) → 2단계(STT) → 3단계(보정) → 4단계(PII탐지) → 5단계(마스킹)
"""

import os
import json
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# privacy-pii 경로 추가
PRIVACY_PII_PATH = Path(__file__).parent / "privacy-pii" / "app"
sys.path.insert(0, str(PRIVACY_PII_PATH))

# 각 단계 모듈 임포트
from audio_enhancer import AudioEnhancer
from speech_to_text import ETRISpeechToText
from conservative_corrector import ConservativeCorrector, InteractiveReviewer
from etri_ner_detector import EnhancedPIIDetector
from detectors.regex_patterns import detect_by_regex
from detectors.keyword_rules import detect_by_keywords
from pii_masker import PIIMasker, MaskingResult


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # API 키
    etri_stt_key: str
    etri_ner_key: str
    openai_key: str
    
    # 각 단계 설정
    enhance_audio: bool = True
    use_context_analysis: bool = True
    use_etri_ner: bool = True
    save_intermediate: bool = True
    
    # 경로
    output_dir: str = "./pipeline_output"
    temp_dir: str = "./temp"


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    session_id: str
    timestamp: str
    
    # 각 단계 결과
    audio_enhancement: Optional[Dict] = None
    speech_to_text: Optional[Dict] = None
    text_correction: Optional[Dict] = None
    pii_detection: Optional[Dict] = None
    masking: Optional[Dict] = None
    
    # 최종 결과
    original_audio_path: str = ""
    enhanced_audio_path: str = ""
    original_text: str = ""
    corrected_text: str = ""
    masked_text: str = ""
    
    # 메타데이터
    processing_time: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class IntegratedPipeline:
    """통합 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 디렉토리 생성
        self.output_dir = Path(config.output_dir) / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 단계 초기화
        self._init_components()
    
    def _init_components(self):
        """컴포넌트 초기화"""
        # 1단계: 음성 향상
        self.audio_enhancer = AudioEnhancer(target_sr=16000)
        
        # 2단계: STT
        self.stt = ETRISpeechToText(self.config.etri_stt_key)
        
        # 3단계: 텍스트 보정
        self.text_corrector = ConservativeCorrector(self.config.openai_key)
        self.text_reviewer = InteractiveReviewer()
        
        # 4단계: PII 탐지
        if self.config.use_etri_ner:
            self.pii_detector = EnhancedPIIDetector(self.config.etri_ner_key)
        else:
            self.pii_detector = None
        
        # 5단계: 마스킹
        self.masker = PIIMasker(
            save_mapping=True,
            mapping_dir=str(self.output_dir / "masking")
        )
    
    def process(self,
                audio_path: str,
                masking_rules: Optional[Dict] = None,
                progress_callback: Optional[callable] = None) -> PipelineResult:
        """
        전체 파이프라인 실행
        
        Args:
            audio_path: 입력 음성 파일
            masking_rules: 마스킹 규칙 (없으면 기본값)
            progress_callback: 진행 상황 콜백 (step, total, message)
        
        Returns:
            PipelineResult
        """
        import time
        
        result = PipelineResult(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            original_audio_path=audio_path
        )
        
        total_steps = 5
        current_step = 0
        
        try:
            # 1단계: 음성 품질 향상
            if self.config.enhance_audio:
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, "🎵 음성 품질 향상 중...")
                
                start_time = time.time()
                enhanced_path = self.output_dir / "enhanced_audio.wav"
                
                enhanced_path, metrics = self.audio_enhancer.enhance(
                    audio_path,
                    str(enhanced_path),
                    visualize=False
                )
                
                result.audio_enhancement = metrics
                result.enhanced_audio_path = enhanced_path
                result.processing_time["audio_enhancement"] = time.time() - start_time
                
                audio_for_stt = enhanced_path
            else:
                audio_for_stt = audio_path
            
            # 2단계: 음성 인식
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "🎤 음성 인식 중...")
            
            start_time = time.time()
            
            stt_result = self.stt.recognize_long_audio(
                audio_for_stt,
                language="korean",
                debug_mode=False
            )
            
            if not stt_result["success"]:
                raise Exception(f"STT 실패: {stt_result.get('error')}")
            
            result.speech_to_text = stt_result
            result.original_text = stt_result["text"]
            result.processing_time["speech_to_text"] = time.time() - start_time
            
            # 3단계: 텍스트 보정
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "✏️ 텍스트 보정 중...")
            
            start_time = time.time()
            
            correction_result = self.text_corrector.correct(
                result.original_text,
                use_context_analysis=self.config.use_context_analysis
            )
            
            result.text_correction = {
                "corrected_text": correction_result.corrected_text,
                "suspicious_parts": len(correction_result.suspicious_parts),
                "auto_corrections": correction_result.auto_corrections,
                "needs_review": correction_result.needs_review
            }
            result.corrected_text = correction_result.corrected_text
            result.processing_time["text_correction"] = time.time() - start_time
            
            # 4단계: PII 탐지
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "🔍 개인정보 탐지 중...")
            
            start_time = time.time()
            
            if self.config.use_etri_ner and self.pii_detector:
                # ETRI NER 사용
                entities = self.pii_detector.detect(
                    result.corrected_text,
                    use_spoken=True  # 구어체 모드
                )
                # entities를 딕셔너리 리스트로 변환
                entities_list = []
                for entity in entities:
                    if hasattr(entity, '__dict__'):
                        entities_list.append(entity.__dict__)
                    elif isinstance(entity, dict):
                        entities_list.append(entity)
                    else:
                        # PIIEntity 객체를 딕셔너리로 변환
                        entities_list.append({
                            "text": entity.text,
                            "label": entity.label,
                            "subtype": entity.subtype,
                            "start": entity.start,
                            "end": entity.end,
                            "score": entity.score,
                            "source": entity.source,
                            "label_adjusted": entity.label
                        })
                entities = entities_list
            else:
                # 정규식 + 키워드만 사용
                regex_hits = detect_by_regex(result.corrected_text)
                entities = detect_by_keywords(result.corrected_text, regex_hits)
            
            result.pii_detection = {
                "total_entities": len(entities),
                "entities": entities
            }
            result.processing_time["pii_detection"] = time.time() - start_time
            
            # 5단계: 마스킹
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "🎭 마스킹 처리 중...")
            
            start_time = time.time()
            
            # 기본 마스킹 규칙 (사용자 지정 없으면)
            if masking_rules is None:
                masking_rules = {
                    "mode": "advanced",
                    "label_rules": {
                        "이름": "[이름]",
                        "번호": "[전화번호]",
                        "주소": "[주소]",
                        "계정": "[이메일]",
                        "금융": "[계좌정보]",
                        "URL": "[링크]",
                        "신원": "[개인정보]",
                        "소속": "[소속]"
                    },
                    "default": "[개인정보]"
                }
            
            # 마스킹 실행
            masking_result = self.masker.mask(
                result.corrected_text,
                entities,  # 이미 딕셔너리 리스트로 변환됨
                masking_rules
            )
            
            # MaskingResult 객체 처리
            if isinstance(masking_result, MaskingResult):
                result.masking = {
                    "masked_text": masking_result.masked_text,
                    "total_masked": masking_result.stats.get("total", 0),
                    "mapping_file": masking_result.mapping_file
                }
                result.masked_text = masking_result.masked_text
            else:
                # 예외 처리
                result.masking = {
                    "masked_text": result.corrected_text,
                    "total_masked": 0,
                    "mapping_file": None
                }
                result.masked_text = result.corrected_text
            
            result.processing_time["masking"] = time.time() - start_time
            
            # 중간 결과 저장
            if self.config.save_intermediate:
                self._save_intermediate_results(result)
            
            if progress_callback:
                progress_callback(total_steps, total_steps, "✅ 완료!")
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"❌ 파이프라인 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _save_intermediate_results(self, result: PipelineResult):
        """중간 결과 저장"""
        # 텍스트 파일들 저장
        texts_dir = self.output_dir / "texts"
        texts_dir.mkdir(exist_ok=True)
        
        # 원본 텍스트
        if result.original_text:
            with open(texts_dir / "1_original.txt", "w", encoding="utf-8") as f:
                f.write(result.original_text)
        
        # 보정된 텍스트
        if result.corrected_text:
            with open(texts_dir / "2_corrected.txt", "w", encoding="utf-8") as f:
                f.write(result.corrected_text)
        
        # 마스킹된 텍스트
        if result.masked_text:
            with open(texts_dir / "3_masked.txt", "w", encoding="utf-8") as f:
                f.write(result.masked_text)
        
        # 전체 결과 JSON
        result_dict = asdict(result)
        with open(self.output_dir / "pipeline_result.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)