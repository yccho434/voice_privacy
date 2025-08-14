# timestamp_audio_pipeline.py
"""
타임스탬프 포함 음성→텍스트 통합 파이프라인
문장별 시작 시간 표시
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# 모듈 임포트
from enhanced_audio_processor import EnhancedAudioProcessor
from smart_sentence_stt import SmartSentenceSTT


@dataclass
class TimestampConfig:
    """파이프라인 설정"""
    etri_api_key: str
    
    # 처리 옵션
    enhance_audio: bool = True
    aggressive_denoise: bool = False
    use_sentence_chunks: bool = True  # 문장 단위 청킹
    parallel_stt: bool = True
    max_workers: int = 2
    
    # 출력 옵션
    output_format: str = "timestamp"  # "timestamp", "srt", "plain"
    save_outputs: bool = True
    output_dir: str = "./output"


@dataclass  
class TimestampResult:
    """처리 결과"""
    success: bool
    
    # 텍스트 결과
    plain_text: str  # 일반 텍스트
    formatted_text: str  # [MM:SS] 포함 텍스트
    srt_text: Optional[str]  # SRT 자막 형식
    
    # 문장 정보
    sentences: List[Dict]  # 각 문장별 정보
    
    # 파일 경로
    audio_path: str
    enhanced_audio_path: Optional[str]
    output_files: Dict[str, str]
    
    # 통계
    total_duration: float
    processing_time: float
    sentence_count: int
    audio_improvement: Dict
    
    timestamp: str


class TimestampAudioPipeline:
    """타임스탬프 음성 처리 파이프라인"""
    
    def __init__(self, config: TimestampConfig):
        self.config = config
        self.audio_processor = EnhancedAudioProcessor(target_sr=16000)
        self.stt_processor = SmartSentenceSTT(config.etri_api_key)
        
        # 출력 디렉토리
        if config.save_outputs:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def process(self,
                audio_path: str,
                progress_callback: Optional[Callable] = None) -> TimestampResult:
        """
        음성 파일 처리 (타임스탬프 포함)
        
        Args:
            audio_path: 입력 음성 파일
            progress_callback: 진행 콜백
            
        Returns:
            TimestampResult
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 진행 상황 관리
        def update_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)
        
        try:
            # 1단계: 음성 품질 향상
            enhanced_audio_path = None
            audio_improvement = {}
            
            if self.config.enhance_audio:
                update_progress(10, "🎵 음성 품질 향상 중...")
                
                if self.config.save_outputs:
                    output_name = f"enhanced_{timestamp}_{Path(audio_path).stem}.wav"
                    enhanced_audio_path = os.path.join(self.config.output_dir, output_name)
                
                audio_data, metrics = self.audio_processor.process(
                    audio_path,
                    enhanced_audio_path,
                    aggressive=self.config.aggressive_denoise
                )
                
                audio_improvement = metrics.get('improvement', {})
                
                # STT용 오디오
                if enhanced_audio_path:
                    audio_for_stt = enhanced_audio_path
                else:
                    import tempfile
                    import soundfile as sf
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    sf.write(temp_file.name, audio_data, 16000)
                    audio_for_stt = temp_file.name
                
                update_progress(30, f"✅ 노이즈 {audio_improvement.get('noise_reduction', 0):.1f}% 감소")
            else:
                audio_for_stt = audio_path
            
            # 2단계: 문장 단위 STT
            update_progress(40, "🎤 문장 단위 음성 인식 시작...")
            
            # STT 진행 콜백
            def stt_progress(current: int, total: int, message: str, eta: Optional[int]):
                base_percent = 40
                stt_range = 50  # 40% ~ 90%
                percent = base_percent + int((current / total) * stt_range)
                update_progress(percent, message)
            
            # 문장 단위 처리
            if self.config.use_sentence_chunks:
                result = self.stt_processor.process_with_timestamps(
                    audio_for_stt,
                    language="korean",
                    max_workers=self.config.max_workers if self.config.parallel_stt else 1,
                    progress_callback=stt_progress
                )
                
                sentences = result.get('sentences', [])
                formatted_text = result.get('formatted_text', '')
                
            else:
                # 폴백: 기존 방식 (타임스탬프 추정)
                from improved_stt_processor import ImprovedSTTProcessor
                fallback_stt = ImprovedSTTProcessor(self.config.etri_api_key)
                result = fallback_stt.process(
                    audio_for_stt,
                    max_workers=self.config.max_workers if self.config.parallel_stt else 1,
                    progress_callback=stt_progress
                )
                
                # 단순 타임스탬프 추정
                text = result.get('text', '')
                sentences = self._estimate_timestamps(text, audio_for_stt)
                formatted_text = self._format_with_timestamps(sentences)
            
            # 3단계: 다양한 형식 생성
            update_progress(90, "📝 결과 포맷팅 중...")
            
            plain_text = self._extract_plain_text(sentences)
            srt_text = self._generate_srt(sentences) if self.config.output_format == "srt" else None
            
            # 4단계: 파일 저장
            output_files = {}
            
            if self.config.save_outputs:
                base_name = f"transcript_{timestamp}_{Path(audio_path).stem}"
                
                # 타임스탬프 포함 텍스트
                if formatted_text:
                    timestamp_file = os.path.join(self.config.output_dir, f"{base_name}_timestamp.txt")
                    with open(timestamp_file, 'w', encoding='utf-8') as f:
                        f.write(formatted_text)
                    output_files['timestamp'] = timestamp_file
                
                # 일반 텍스트
                plain_file = os.path.join(self.config.output_dir, f"{base_name}_plain.txt")
                with open(plain_file, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                output_files['plain'] = plain_file
                
                # SRT 자막
                if srt_text:
                    srt_file = os.path.join(self.config.output_dir, f"{base_name}.srt")
                    with open(srt_file, 'w', encoding='utf-8') as f:
                        f.write(srt_text)
                    output_files['srt'] = srt_file
                
                # JSON 메타데이터
                meta_file = os.path.join(self.config.output_dir, f"{base_name}_meta.json")
                meta_data = {
                    'timestamp': timestamp,
                    'audio_path': audio_path,
                    'sentences': sentences,
                    'audio_improvement': audio_improvement,
                    'processing_time': time.time() - start_time
                }
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
                output_files['meta'] = meta_file
            
            # 오디오 길이 계산
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=None)
                total_duration = len(y) / sr
            except:
                total_duration = sentences[-1]['end_time'] if sentences else 0
            
            # 임시 파일 정리
            if self.config.enhance_audio and not self.config.save_outputs:
                try:
                    os.remove(audio_for_stt)
                except:
                    pass
            
            update_progress(100, "✅ 처리 완료!")
            
            return TimestampResult(
                success=True,
                plain_text=plain_text,
                formatted_text=formatted_text,
                srt_text=srt_text,
                sentences=sentences,
                audio_path=audio_path,
                enhanced_audio_path=enhanced_audio_path,
                output_files=output_files,
                total_duration=total_duration,
                processing_time=time.time() - start_time,
                sentence_count=len(sentences),
                audio_improvement=audio_improvement,
                timestamp=timestamp
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            return TimestampResult(
                success=False,
                plain_text="",
                formatted_text="",
                srt_text=None,
                sentences=[],
                audio_path=audio_path,
                enhanced_audio_path=None,
                output_files={},
                total_duration=0,
                processing_time=time.time() - start_time,
                sentence_count=0,
                audio_improvement={},
                timestamp=timestamp
            )
    
    def _extract_plain_text(self, sentences: List[Dict]) -> str:
        """일반 텍스트 추출"""
        return " ".join(s['text'] for s in sentences if s.get('text'))
    
    def _format_with_timestamps(self, sentences: List[Dict]) -> str:
        """타임스탬프 포맷팅"""
        lines = []
        
        for sentence in sentences:
            minutes = int(sentence['start_time'] // 60)
            seconds = int(sentence['start_time'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            lines.append(f"{timestamp} {sentence['text']}")
            lines.append("")  # 빈 줄
        
        return "\n".join(lines)
    
    def _generate_srt(self, sentences: List[Dict]) -> str:
        """SRT 자막 형식 생성"""
        srt_lines = []
        
        for i, sentence in enumerate(sentences, 1):
            # 시간 포맷 (HH:MM:SS,mmm)
            start_time = self._format_srt_time(sentence['start_time'])
            end_time = self._format_srt_time(sentence.get('end_time', sentence['start_time'] + 3))
            
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(sentence['text'])
            srt_lines.append("")  # 빈 줄
        
        return "\n".join(srt_lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """SRT 시간 형식 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _estimate_timestamps(self, text: str, audio_path: str) -> List[Dict]:
        """폴백: 단순 타임스탬프 추정"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            total_duration = len(y) / sr
        except:
            total_duration = 60  # 기본값
        
        # 문장 분리
        import re
        sentences_text = re.split(r'[.!?]+', text)
        sentences_text = [s.strip() for s in sentences_text if s.strip()]
        
        if not sentences_text:
            return []
        
        # 균등 분배
        time_per_sentence = total_duration / len(sentences_text)
        
        sentences = []
        current_time = 0
        
        for sent_text in sentences_text:
            sentences.append({
                'text': sent_text,
                'start_time': current_time,
                'end_time': current_time + time_per_sentence
            })
            current_time += time_per_sentence
        
        return sentences


# 간편 사용 함수
def process_audio_with_timestamps(
    audio_path: str,
    api_key: str,
    enhance_audio: bool = True,
    output_format: str = "timestamp",
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    타임스탬프 포함 음성 처리
    
    Args:
        audio_path: 오디오 파일
        api_key: ETRI API 키
        enhance_audio: 음성 향상 여부
        output_format: 출력 형식 ("timestamp", "srt", "plain")
        progress_callback: 진행 콜백
        
    Returns:
        처리 결과 딕셔너리
    """
    config = TimestampConfig(
        etri_api_key=api_key,
        enhance_audio=enhance_audio,
        output_format=output_format,
        use_sentence_chunks=True  # 문장 단위 청킹 사용
    )
    
    pipeline = TimestampAudioPipeline(config)
    result = pipeline.process(audio_path, progress_callback)
    
    return {
        'success': result.success,
        'formatted_text': result.formatted_text,
        'plain_text': result.plain_text,
        'sentences': result.sentences,
        'sentence_count': result.sentence_count,
        'duration': result.total_duration,
        'processing_time': result.processing_time,
        'files': result.output_files
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python timestamp_audio_pipeline.py [오디오파일]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    api_key = os.getenv("ETRI_API_KEY")
    
    if not api_key:
        print("❌ ETRI API 키 설정 필요")
        sys.exit(1)
    
    print(f"🎵 처리 시작: {audio_file}\n")
    
    # 진행 표시
    def show_progress(percent, message):
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {percent:3d}% | {message}", end="", flush=True)
    
    # 처리
    result = process_audio_with_timestamps(
        audio_file,
        api_key,
        enhance_audio=True,
        output_format="timestamp",
        progress_callback=show_progress
    )
    
    print("\n\n" + "="*60)
    
    if result['success']:
        print("✅ 처리 완료!\n")
        print(f"📊 통계:")
        print(f"  - 문장 수: {result['sentence_count']}개")
        print(f"  - 오디오 길이: {result['duration']:.1f}초")
        print(f"  - 처리 시간: {result['processing_time']:.1f}초")
        print(f"\n📝 타임스탬프 포함 텍스트:\n")
        print(result['formatted_text'][:1000])
        if len(result['formatted_text']) > 1000:
            print("\n... (이하 생략)")
    else:
        print("❌ 처리 실패")