# audio_to_text_pipeline.py
"""
통합 음성→텍스트 파이프라인
librosa 전처리 + ETRI STT 통합
UX 최적화 버전
"""

import os
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# 모듈 임포트
from enhanced_audio_processor import EnhancedAudioProcessor
from improved_stt_processor import ImprovedSTTProcessor


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # API 설정
    etri_api_key: str
    
    # 처리 옵션
    enhance_audio: bool = True
    aggressive_denoise: bool = False
    auto_detect_noise: bool = True
    parallel_stt: bool = True
    max_workers: int = 2
    enable_timestamps: bool = True  # 타임스탬프 옵션 추가
    
    # 출력 설정
    save_enhanced_audio: bool = False
    save_transcript: bool = True
    output_dir: str = "./output"


@dataclass
class PipelineResult:
    """파이프라인 결과"""
    success: bool
    transcript: str
    formatted_transcript: Optional[str]  # 타임스탬프 포함 텍스트
    sentences: Optional[list[Dict]]  # 문장별 정보
    
    # 처리 정보
    original_audio_path: str
    enhanced_audio_path: Optional[str]
    transcript_path: Optional[str]
    
    # 통계
    audio_duration: float
    processing_time: float
    audio_improvement: Dict
    stt_stats: Dict
    
    # 타임스탬프
    timestamp: str
    

class AudioToTextPipeline:
    """통합 음성→텍스트 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.audio_processor = EnhancedAudioProcessor(target_sr=16000)
        self.stt_processor = ImprovedSTTProcessor(config.etri_api_key)
        
        # 출력 디렉토리 생성
        if config.save_enhanced_audio or config.save_transcript:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def process(self,
                audio_path: str,
                progress_callback: Optional[Callable] = None) -> PipelineResult:
        """
        음성 파일을 텍스트로 변환
        
        Args:
            audio_path: 입력 음성 파일
            progress_callback: 진행 콜백 (step, percent, message, eta_seconds)
        
        Returns:
            PipelineResult
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 진행 상황 추적
        total_steps = 2 if self.config.enhance_audio else 1
        current_step = 0
        
        def update_progress(percent: int, message: str, eta: Optional[int] = None):
            """내부 진행 콜백"""
            if progress_callback:
                # 전체 진행률 계산
                step_weight = 100 / total_steps
                overall_percent = int(current_step * step_weight + percent * step_weight / 100)
                
                # ETA 포맷팅
                eta_str = None
                if eta:
                    eta_str = str(timedelta(seconds=eta))
                
                progress_callback(current_step + 1, overall_percent, message, eta_str)
        
        try:
            # 1단계: 음성 품질 향상 (선택적)
            audio_improvement = {}
            enhanced_audio_path = None
            
            if self.config.enhance_audio:
                current_step = 0
                update_progress(0, "🎵 음성 품질 향상 중...")
                
                # 향상 처리
                if self.config.save_enhanced_audio:
                    output_name = f"enhanced_{timestamp}_{Path(audio_path).stem}.wav"
                    enhanced_audio_path = os.path.join(self.config.output_dir, output_name)
                else:
                    enhanced_audio_path = None
                
                audio_data, metrics = self.audio_processor.process(
                    audio_path,
                    enhanced_audio_path,
                    aggressive=self.config.aggressive_denoise,
                    auto_detect_noise=self.config.auto_detect_noise
                )
                
                audio_improvement = metrics.get('improvement', {})
                
                # 임시 파일 생성 (STT용)
                if not enhanced_audio_path:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    import soundfile as sf
                    sf.write(temp_file.name, audio_data, 16000)
                    audio_for_stt = temp_file.name
                else:
                    audio_for_stt = enhanced_audio_path
                
                update_progress(100, "✅ 음성 향상 완료")
                
                # 품질 개선 요약
                improvement_msg = f"노이즈 {audio_improvement.get('noise_reduction', 0):.1f}% 감소"
                update_progress(100, improvement_msg)
                
            else:
                audio_for_stt = audio_path
            
            # 2단계: 음성 인식
            current_step = 1 if self.config.enhance_audio else 0
            update_progress(0, "🎤 음성 인식 시작...")
            
            # STT 진행 콜백 래퍼
            def stt_progress(current: int, total: int, message: str, eta: Optional[int]):
                percent = int((current / total) * 100) if total > 0 else 0
                update_progress(percent, f"🎤 {message}", eta)
            
            # STT 실행
            max_workers = self.config.max_workers if self.config.parallel_stt else 1
            stt_result = self.stt_processor.process(
                audio_for_stt,
                language="korean",
                max_workers=max_workers,
                progress_callback=stt_progress,
                enable_timestamps=self.config.enable_timestamps  # 타임스탬프 옵션 전달
            )
            
            # 임시 파일 정리
            if self.config.enhance_audio and not self.config.save_enhanced_audio:
                try:
                    os.remove(audio_for_stt)
                except:
                    pass
            
            # 3단계: 결과 저장
            transcript_path = None
            formatted_transcript_path = None
            
            if self.config.save_transcript and stt_result.get('text'):
                # 일반 텍스트 저장
                transcript_name = f"transcript_{timestamp}_{Path(audio_path).stem}.txt"
                transcript_path = os.path.join(self.config.output_dir, transcript_name)
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(stt_result['text'])
                
                # 타임스탬프 포함 텍스트 저장
                if self.config.enable_timestamps and stt_result.get('formatted_text'):
                    formatted_name = f"transcript_{timestamp}_{Path(audio_path).stem}_timestamp.txt"
                    formatted_transcript_path = os.path.join(self.config.output_dir, formatted_name)
                    
                    with open(formatted_transcript_path, 'w', encoding='utf-8') as f:
                        f.write(stt_result['formatted_text'])
                
                # 메타데이터도 저장
                meta_path = transcript_path.replace('.txt', '_meta.json')
                meta_data = {
                    'timestamp': timestamp,
                    'original_audio': audio_path,
                    'audio_improvement': audio_improvement,
                    'stt_stats': stt_result.get('stats', {}),
                    'processing_time': time.time() - start_time,
                    'sentences': stt_result.get('sentences', []) if self.config.enable_timestamps else []
                }
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            # 완료
            update_progress(100, "✅ 처리 완료!")
            
            # 오디오 길이 계산
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=None)
                audio_duration = len(y) / sr
            except:
                audio_duration = 0
            
            # 결과 생성
            return PipelineResult(
                success=stt_result.get('success', False),
                transcript=stt_result.get('text', ''),
                formatted_transcript=stt_result.get('formatted_text'),  # 타임스탬프 포함
                sentences=stt_result.get('sentences'),  # 문장별 정보
                original_audio_path=audio_path,
                enhanced_audio_path=enhanced_audio_path,
                transcript_path=transcript_path,
                audio_duration=audio_duration,
                processing_time=time.time() - start_time,
                audio_improvement=audio_improvement,
                stt_stats=stt_result.get('stats', {}),
                timestamp=timestamp
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            return PipelineResult(
                success=False,
                transcript='',
                formatted_transcript=None,
                sentences=None,
                original_audio_path=audio_path,
                enhanced_audio_path=None,
                transcript_path=None,
                audio_duration=0,
                processing_time=time.time() - start_time,
                audio_improvement={},
                stt_stats={'error': str(e)},
                timestamp=timestamp
            )
    
    def process_batch(self,
                     audio_files: list,
                     progress_callback: Optional[Callable] = None) -> list:
        """
        여러 파일 일괄 처리
        
        Args:
            audio_files: 오디오 파일 경로 리스트
            progress_callback: 진행 콜백
        
        Returns:
            PipelineResult 리스트
        """
        results = []
        total_files = len(audio_files)
        
        for i, audio_path in enumerate(audio_files):
            # 파일별 진행 콜백
            def file_progress(step: int, percent: int, message: str, eta: str):
                if progress_callback:
                    overall_percent = int((i / total_files) * 100 + percent / total_files)
                    file_info = f"[{i+1}/{total_files}] {Path(audio_path).name}"
                    progress_callback(i+1, overall_percent, f"{file_info}: {message}", eta)
            
            # 처리
            result = self.process(audio_path, file_progress)
            results.append(result)
            
            # 성공/실패 로그
            if result.success:
                print(f"✅ {Path(audio_path).name}: {len(result.transcript)}자 변환 완료")
            else:
                print(f"❌ {Path(audio_path).name}: 변환 실패")
        
        return results


def process_audio_to_text(
    audio_path: str,
    etri_api_key: str,
    enhance_audio: bool = True,
    save_outputs: bool = True,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    간편 실행 함수
    
    Args:
        audio_path: 오디오 파일 경로
        etri_api_key: ETRI API 키
        enhance_audio: 음성 향상 여부
        save_outputs: 결과 저장 여부
        progress_callback: 진행 콜백
    
    Returns:
        결과 딕셔너리
    """
    config = PipelineConfig(
        etri_api_key=etri_api_key,
        enhance_audio=enhance_audio,
        save_enhanced_audio=save_outputs,
        save_transcript=save_outputs
    )
    
    pipeline = AudioToTextPipeline(config)
    result = pipeline.process(audio_path, progress_callback)
    
    return {
        'success': result.success,
        'transcript': result.transcript,
        'processing_time': result.processing_time,
        'audio_duration': result.audio_duration,
        'improvement': result.audio_improvement,
        'stats': result.stt_stats,
        'files': {
            'enhanced_audio': result.enhanced_audio_path,
            'transcript': result.transcript_path
        }
    }


if __name__ == "__main__":
    # 테스트 실행
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python audio_to_text_pipeline.py [오디오파일]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    api_key = os.getenv("ETRI_API_KEY")
    
    if not api_key:
        print("❌ ETRI API 키를 설정하세요: export ETRI_API_KEY='your_key'")
        sys.exit(1)
    
    print(f"🎵 처리 시작: {audio_file}")
    
    # 진행 표시
    def show_progress(step, percent, message, eta):
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        eta_str = f" ETA: {eta}" if eta else ""
        print(f"\r[{bar}] {percent:3d}% | {message}{eta_str}", end="", flush=True)
    
    # 처리 실행
    result = process_audio_to_text(
        audio_file,
        api_key,
        enhance_audio=True,
        save_outputs=True,
        progress_callback=show_progress
    )
    
    print("\n")
    
    if result['success']:
        print("✅ 처리 완료!")
        print(f"- 처리 시간: {result['processing_time']:.1f}초")
        print(f"- 오디오 길이: {result['audio_duration']:.1f}초")
        print(f"- 텍스트 길이: {len(result['transcript'])}자")
        print(f"- 노이즈 감소: {result['improvement'].get('noise_reduction', 0):.1f}%")
        print(f"\n📝 변환 텍스트:\n{result['transcript'][:500]}...")
    else:
        print("❌ 처리 실패")
        print(result.get('stats', {}).get('error', 'Unknown error'))