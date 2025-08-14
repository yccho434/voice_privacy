# smart_sentence_stt.py
"""
문장 단위 스마트 STT 처리기
정확한 타임스탬프를 위한 문장별 청킹
"""

import json
import base64
import urllib3
import time
import tempfile
import os
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# pydub 설정
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent, split_on_silence
    import imageio_ffmpeg
    
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub 설치 필요")


@dataclass
class SentenceChunk:
    """문장 청크 정보"""
    index: int
    audio_path: str
    start_time: float  # 초 단위
    end_time: float
    duration: float
    estimated_text_length: int  # 예상 텍스트 길이
    is_merged: bool = False  # 다른 청크와 병합됨
    merged_indices: List[int] = None  # 병합된 청크 인덱스들


@dataclass
class SentenceResult:
    """문장 처리 결과"""
    chunk_index: int
    text: str
    start_time: float
    end_time: float
    success: bool
    error: Optional[str] = None


class SmartSentenceSTT:
    """문장 단위 스마트 STT 처리기"""
    
    API_URL = "http://epretx.etri.re.kr:8000/api/WiseASR_Recognition"
    
    # 청크 설정
    MIN_CHUNK_SEC = 3.0   # 최소 3초
    MAX_CHUNK_SEC = 15.0  # 최대 15초 (API 안전 마진)
    OPTIMAL_CHUNK_SEC = 8.0  # 최적 8초
    
    # 무음 감지 설정
    SILENCE_THRESH = -35  # dB
    MIN_SILENCE_LEN = 300  # ms (문장 경계)
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http = urllib3.PoolManager()
        self.sentence_boundaries = []  # 문장 경계 시간 저장
    
    def process_with_timestamps(self,
                               audio_path: str,
                               language: str = "korean",
                               max_workers: int = 2,
                               progress_callback: Optional[Callable] = None) -> Dict:
        """
        타임스탬프 포함 STT 처리
        
        Returns:
            {
                'success': bool,
                'sentences': [
                    {
                        'text': str,
                        'start_time': float,
                        'end_time': float
                    }
                ],
                'formatted_text': str  # [MM:SS] 형식 포함
            }
        """
        start_time = time.time()
        
        # 1. 문장 단위 청크 생성
        if progress_callback:
            progress_callback(0, 100, "음성 분석 및 문장 경계 감지 중...", None)
        
        chunks = self._create_sentence_chunks(audio_path)
        
        if not chunks:
            return {
                'success': False,
                'sentences': [],
                'formatted_text': '',
                'error': '문장 청크 생성 실패'
            }
        
        print(f"📦 {len(chunks)}개 문장 청크 생성됨")
        
        # 2. 청크 처리 (병렬 또는 순차)
        if len(chunks) <= 3 or max_workers == 1:
            results = self._process_sequential(chunks, language, progress_callback)
        else:
            results = self._process_parallel(chunks, language, max_workers, progress_callback)
        
        # 3. 결과 정리 및 포맷팅
        sentences = self._organize_results(results, chunks)
        formatted_text = self._format_with_timestamps(sentences)
        
        # 통계
        success_count = sum(1 for r in results if r.success)
        total_time = time.time() - start_time
        
        # 임시 파일 정리
        for chunk in chunks:
            try:
                os.remove(chunk.audio_path)
            except:
                pass
        
        return {
            'success': success_count > len(chunks) * 0.5,
            'sentences': sentences,
            'formatted_text': formatted_text,
            'stats': {
                'total_chunks': len(chunks),
                'success_chunks': success_count,
                'processing_time': total_time
            }
        }
    
    def _create_sentence_chunks(self, audio_path: str) -> List[SentenceChunk]:
        """문장 단위로 오디오 분할"""
        if not PYDUB_AVAILABLE:
            return []
        
        audio = AudioSegment.from_file(audio_path)
        total_duration_ms = len(audio)
        
        # 1. 무음 기반 분할
        chunks_raw = split_on_silence(
            audio,
            min_silence_len=self.MIN_SILENCE_LEN,
            silence_thresh=self.SILENCE_THRESH,
            keep_silence=200  # 앞뒤 200ms 유지 (자연스러운 발화)
        )
        
        if not chunks_raw:
            # 무음이 없으면 고정 길이로 분할
            return self._fallback_fixed_chunks(audio)
        
        # 2. 청크 정보 생성
        temp_dir = tempfile.mkdtemp()
        chunks = []
        current_position = 0
        
        for i, chunk_audio in enumerate(chunks_raw):
            chunk_duration_ms = len(chunk_audio)
            
            # 시작/끝 시간 계산
            # 실제 오디오에서의 위치 찾기 (근사치)
            start_ms = current_position
            end_ms = start_ms + chunk_duration_ms
            
            # 청크 정보 생성
            chunk_info = SentenceChunk(
                index=i,
                audio_path="",  # 나중에 설정
                start_time=start_ms / 1000.0,
                end_time=end_ms / 1000.0,
                duration=chunk_duration_ms / 1000.0,
                estimated_text_length=self._estimate_text_length(chunk_duration_ms)
            )
            
            chunks.append((chunk_info, chunk_audio))
            current_position = end_ms
        
        # 3. 청크 최적화 (병합/분할)
        optimized_chunks = self._optimize_chunks(chunks)
        
        # 4. 오디오 파일 저장
        final_chunks = []
        for chunk_info, chunk_audio in optimized_chunks:
            chunk_path = os.path.join(temp_dir, f"sentence_{chunk_info.index:04d}.wav")
            chunk_audio.export(chunk_path, format="wav")
            chunk_info.audio_path = chunk_path
            final_chunks.append(chunk_info)
        
        return final_chunks
    
    def _optimize_chunks(self, 
                        chunks: List[Tuple[SentenceChunk, AudioSegment]]) -> List[Tuple[SentenceChunk, AudioSegment]]:
        """청크 최적화 (병합/분할)"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            chunk_info, chunk_audio = chunks[i]
            
            # 너무 짧은 청크: 다음과 병합
            if chunk_info.duration < self.MIN_CHUNK_SEC and i < len(chunks) - 1:
                # 다음 청크들과 병합
                merged_audio = chunk_audio
                merged_info = SentenceChunk(
                    index=chunk_info.index,
                    audio_path="",
                    start_time=chunk_info.start_time,
                    end_time=chunk_info.end_time,
                    duration=chunk_info.duration,
                    estimated_text_length=chunk_info.estimated_text_length,
                    is_merged=True,
                    merged_indices=[chunk_info.index]
                )
                
                j = i + 1
                while j < len(chunks) and merged_info.duration < self.OPTIMAL_CHUNK_SEC:
                    next_info, next_audio = chunks[j]
                    
                    # 병합 후 너무 길어지면 중단
                    if merged_info.duration + next_info.duration > self.MAX_CHUNK_SEC:
                        break
                    
                    merged_audio += next_audio
                    merged_info.end_time = next_info.end_time
                    merged_info.duration = merged_info.end_time - merged_info.start_time
                    merged_info.estimated_text_length += next_info.estimated_text_length
                    merged_info.merged_indices.append(next_info.index)
                    j += 1
                
                optimized.append((merged_info, merged_audio))
                i = j
            
            # 너무 긴 청크: 분할
            elif chunk_info.duration > self.MAX_CHUNK_SEC:
                # 중간 지점에서 분할
                split_point = len(chunk_audio) // 2
                
                first_audio = chunk_audio[:split_point]
                second_audio = chunk_audio[split_point:]
                
                first_info = SentenceChunk(
                    index=chunk_info.index,
                    audio_path="",
                    start_time=chunk_info.start_time,
                    end_time=chunk_info.start_time + len(first_audio) / 1000.0,
                    duration=len(first_audio) / 1000.0,
                    estimated_text_length=chunk_info.estimated_text_length // 2
                )
                
                second_info = SentenceChunk(
                    index=chunk_info.index + 0.5,  # 소수점으로 구분
                    audio_path="",
                    start_time=first_info.end_time,
                    end_time=chunk_info.end_time,
                    duration=len(second_audio) / 1000.0,
                    estimated_text_length=chunk_info.estimated_text_length // 2
                )
                
                optimized.append((first_info, first_audio))
                optimized.append((second_info, second_audio))
                i += 1
            
            # 적절한 길이: 그대로 사용
            else:
                optimized.append((chunk_info, chunk_audio))
                i += 1
        
        print(f"📊 청크 최적화: {len(chunks)}개 → {len(optimized)}개")
        return optimized
    
    def _fallback_fixed_chunks(self, audio: AudioSegment) -> List[SentenceChunk]:
        """무음 감지 실패 시 고정 길이 분할"""
        chunks = []
        temp_dir = tempfile.mkdtemp()
        
        chunk_ms = int(self.OPTIMAL_CHUNK_SEC * 1000)
        total_ms = len(audio)
        
        for i in range(0, total_ms, chunk_ms):
            start_ms = i
            end_ms = min(i + chunk_ms, total_ms)
            
            chunk_audio = audio[start_ms:end_ms]
            chunk_path = os.path.join(temp_dir, f"fixed_{i//chunk_ms:04d}.wav")
            chunk_audio.export(chunk_path, format="wav")
            
            chunks.append(SentenceChunk(
                index=i // chunk_ms,
                audio_path=chunk_path,
                start_time=start_ms / 1000.0,
                end_time=end_ms / 1000.0,
                duration=(end_ms - start_ms) / 1000.0,
                estimated_text_length=100  # 추정값
            ))
        
        return chunks
    
    def _estimate_text_length(self, duration_ms: int) -> int:
        """음성 길이로 텍스트 길이 추정"""
        # 한국어 평균: 분당 200자 정도
        chars_per_second = 200 / 60
        return int((duration_ms / 1000) * chars_per_second)
    
    def _api_call(self, chunk_path: str, language: str) -> SentenceResult:
        """API 호출"""
        try:
            with open(chunk_path, "rb") as f:
                audio_contents = base64.b64encode(f.read()).decode("utf8")
            
            request_json = {
                "argument": {
                    "language_code": language,
                    "audio": audio_contents
                }
            }
            
            response = self.http.request(
                "POST",
                self.API_URL,
                headers={
                    "Content-Type": "application/json; charset=UTF-8",
                    "Authorization": self.api_key
                },
                body=json.dumps(request_json),
                timeout=30.0
            )
            
            if response.status == 200:
                result = json.loads(response.data.decode("utf-8"))
                if result.get("result", -1) == 0:
                    text = result.get("return_object", {}).get("recognized", "")
                    return SentenceResult(
                        chunk_index=0,
                        text=text,
                        start_time=0,
                        end_time=0,
                        success=True
                    )
            
            return SentenceResult(
                chunk_index=0,
                text="",
                start_time=0,
                end_time=0,
                success=False,
                error=f"API Error: {response.status}"
            )
            
        except Exception as e:
            return SentenceResult(
                chunk_index=0,
                text="",
                start_time=0,
                end_time=0,
                success=False,
                error=str(e)
            )
    
    def _process_sequential(self, chunks: List[SentenceChunk], language: str,
                          progress_callback: Optional[Callable]) -> List[SentenceResult]:
        """순차 처리"""
        results = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                percent = int((i / total) * 100)
                progress_callback(percent, 100, f"문장 {i+1}/{total} 처리 중...", None)
            
            result = self._api_call(chunk.audio_path, language)
            result.chunk_index = chunk.index
            result.start_time = chunk.start_time
            result.end_time = chunk.end_time
            results.append(result)
            
            if i < total - 1:
                time.sleep(1.0)
        
        return results
    
    def _process_parallel(self, chunks: List[SentenceChunk], language: str,
                        max_workers: int, progress_callback: Optional[Callable]) -> List[SentenceResult]:
        """병렬 처리"""
        results = []
        completed = 0
        total = len(chunks)
        
        with ThreadPoolExecutor(max_workers=min(max_workers, 2)) as executor:
            future_to_chunk = {}
            
            for i, chunk in enumerate(chunks):
                if i > 0 and i % 2 == 0:
                    time.sleep(0.5)  # API 부하 관리
                
                future = executor.submit(self._api_call, chunk.audio_path, language)
                future_to_chunk[future] = chunk
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                
                try:
                    result = future.result(timeout=30)
                    result.chunk_index = chunk.index
                    result.start_time = chunk.start_time
                    result.end_time = chunk.end_time
                    results.append(result)
                    
                except Exception as e:
                    results.append(SentenceResult(
                        chunk_index=chunk.index,
                        text="",
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        success=False,
                        error=str(e)
                    ))
                
                completed += 1
                if progress_callback:
                    percent = int((completed / total) * 100)
                    progress_callback(percent, 100, f"병렬 처리 중... {completed}/{total}", None)
        
        # 시간 순서로 정렬
        results.sort(key=lambda x: x.start_time)
        return results
    
    def _organize_results(self, results: List[SentenceResult], 
                         chunks: List[SentenceChunk]) -> List[Dict]:
        """결과를 문장 단위로 정리"""
        sentences = []
        
        for result in results:
            if not result.success or not result.text:
                continue
            
            # 병합된 청크 처리
            chunk = next((c for c in chunks if c.index == result.chunk_index), None)
            if chunk and chunk.is_merged and chunk.merged_indices:
                # 병합된 문장들을 분리
                text_sentences = self._split_text_to_sentences(result.text)
                
                if len(text_sentences) == 1:
                    # 분리 안 됨: 전체를 하나로
                    sentences.append({
                        'text': result.text.strip(),
                        'start_time': result.start_time,
                        'end_time': result.end_time
                    })
                else:
                    # 시간을 비례 배분
                    total_duration = result.end_time - result.start_time
                    current_time = result.start_time
                    
                    for sent_text in text_sentences:
                        # 텍스트 길이 비례로 시간 할당
                        sent_duration = (len(sent_text) / len(result.text)) * total_duration
                        
                        sentences.append({
                            'text': sent_text.strip(),
                            'start_time': current_time,
                            'end_time': current_time + sent_duration
                        })
                        
                        current_time += sent_duration
            else:
                # 단일 문장
                sentences.append({
                    'text': result.text.strip(),
                    'start_time': result.start_time,
                    'end_time': result.end_time
                })
        
        return sentences
    
    def _split_text_to_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        # 간단한 문장 분리 (향후 개선 가능)
        import re
        
        # 문장 종결 패턴
        sentences = re.split(r'([.!?]+)\s*', text)
        
        # 종결 부호를 문장에 다시 붙이기
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        
        # 빈 문장 제거
        result = [s.strip() for s in result if s.strip()]
        
        return result if result else [text]
    
    def _format_with_timestamps(self, sentences: List[Dict]) -> str:
        """타임스탬프 포함 포맷팅"""
        formatted_lines = []
        
        for sentence in sentences:
            # 시간 포맷 (MM:SS)
            minutes = int(sentence['start_time'] // 60)
            seconds = int(sentence['start_time'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            # 문장 추가
            formatted_lines.append(f"{timestamp} {sentence['text']}")
            formatted_lines.append("")  # 빈 줄 추가 (가독성)
        
        return "\n".join(formatted_lines)


# 간편 사용 함수
def transcribe_with_timestamps(audio_path: str, api_key: str,
                              max_workers: int = 2,
                              progress_callback: Optional[Callable] = None) -> Dict:
    """
    타임스탬프 포함 음성 인식
    
    Returns:
        {
            'success': bool,
            'formatted_text': str,  # [MM:SS] 형식
            'sentences': list,  # 문장별 정보
            'stats': dict
        }
    """
    processor = SmartSentenceSTT(api_key)
    return processor.process_with_timestamps(
        audio_path, 
        max_workers=max_workers,
        progress_callback=progress_callback
    )