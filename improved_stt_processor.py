# improved_stt_processor.py
"""
개선된 ETRI STT 처리 모듈 (타임스탬프 통합 버전)
- 문장 단위 스마트 청킹으로 정확한 타임스탬프
- 20초 제한 내 최적화
- 안정적인 병렬 처리 (최대 2워커)
"""

import json
import base64
import urllib3
import time
import tempfile
import os
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# pydub 자동 설정
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent, detect_silence, split_on_silence
    import imageio_ffmpeg
    
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub 설치 필요")


@dataclass
class ChunkInfo:
    """청크 정보"""
    index: int
    path: str
    start_ms: int
    end_ms: int
    duration_ms: int
    has_speech: bool
    overlap_with_next: int = 0
    is_merged: bool = False
    merged_indices: List[int] = None
    
@dataclass
class ProcessResult:
    """처리 결과"""
    success: bool
    text: str
    chunk_index: int
    duration: float
    retry_count: int = 0
    error: Optional[str] = None
    start_time: float = 0.0  # 타임스탬프 추가
    end_time: float = 0.0


class ImprovedSTTProcessor:
    """개선된 STT 처리기"""
    
    API_URL = "http://epretx.etri.re.kr:8000/api/WiseASR_Recognition"
    MAX_DURATION_MS = 20000  # 20초 (API 제한)
    MIN_CHUNK_MS = 3000      # 최소 3초
    OPTIMAL_CHUNK_MS = 15000 # 최적 15초
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http = urllib3.PoolManager()
        self.processing_stats = {
            'total_chunks': 0,
            'success_chunks': 0,
            'failed_chunks': 0,
            'retry_count': 0,
            'total_duration': 0
        }
    
    def process(self,
                audio_path: str,
                language: str = "korean",
                max_workers: int = 2,
                progress_callback: Optional[Callable] = None,
                enable_timestamps: bool = True) -> Dict:
        """
        메인 STT 처리 함수 (타임스탬프 옵션 추가)
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드
            max_workers: 병렬 워커 수 (1-2)
            progress_callback: 진행 콜백(current, total, message, eta)
            enable_timestamps: 타임스탬프 포함 여부
        
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        
        # 1. 오디오 분석 및 청크 생성
        if progress_callback:
            progress_callback(0, 100, "오디오 분석 중...", None)
        
        # 타임스탬프 모드면 문장 단위 청킹
        if enable_timestamps:
            chunks = self._create_sentence_chunks(audio_path)
        else:
            chunks = self._create_smart_chunks(audio_path)
            
        if not chunks:
            return {
                'success': False,
                'text': '',
                'error': '청크 생성 실패'
            }
        
        self.processing_stats['total_chunks'] = len(chunks)
        
        # 2. 처리 모드 결정
        if len(chunks) == 1:
            mode = "single"
        elif max_workers == 1 or len(chunks) <= 2:
            mode = "sequential"
        else:
            mode = "parallel"
        
        # 3. 청크 처리
        if mode == "single":
            results = self._process_single(chunks[0], language, progress_callback)
        elif mode == "sequential":
            results = self._process_sequential(chunks, language, progress_callback)
        else:
            results = self._process_parallel(chunks, language, max_workers, progress_callback)
        
        # 4. 타임스탬프 처리
        if enable_timestamps:
            sentences = self._organize_with_timestamps(results, chunks)
            final_text = self._extract_plain_text(sentences)
            formatted_text = self._format_with_timestamps(sentences)
        else:
            # 기존 방식
            final_text = self._merge_results(results)
            formatted_text = None
            sentences = None
        
        # 5. 통계 및 결과 생성
        total_time = time.time() - start_time
        success_rate = self.processing_stats['success_chunks'] / self.processing_stats['total_chunks']
        
        # 청크 정리
        for chunk in chunks:
            try:
                os.remove(chunk.path)
            except:
                pass
        
        result_dict = {
            'success': success_rate > 0.5,
            'text': final_text,
            'mode': mode,
            'stats': {
                **self.processing_stats,
                'success_rate': success_rate,
                'processing_time': total_time
            },
            'chunks_detail': [
                {
                    'index': r.chunk_index,
                    'success': r.success,
                    'text_length': len(r.text) if r.success else 0,
                    'retry_count': r.retry_count
                } for r in results
            ]
        }
        
        # 타임스탬프 정보 추가
        if enable_timestamps:
            result_dict['formatted_text'] = formatted_text
            result_dict['sentences'] = sentences
        
        return result_dict
    
    def _create_sentence_chunks(self, audio_path: str) -> List[ChunkInfo]:
        """문장 단위 스마트 청킹 (정확한 타임스탬프)"""
        if not PYDUB_AVAILABLE:
            return []
        
        from pydub.silence import detect_nonsilent
        
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        
        # 무음이 아닌 구간 감지 (실제 발화 위치)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=300,  # 0.3초 이상 무음
            silence_thresh=-35,   # -35dB
            seek_step=10
        )
        
        if not nonsilent_ranges:
            # 무음만 있는 경우
            return []
        
        print(f"📊 발화 구간 감지: {len(nonsilent_ranges)}개")
        
        # 청크 정보 생성
        temp_dir = tempfile.mkdtemp()
        chunks = []
        
        # 발화 구간을 청크로 변환 (실제 시간 위치 보존)
        optimized_chunks = []
        i = 0
        
        while i < len(nonsilent_ranges):
            start_ms, end_ms = nonsilent_ranges[i]
            chunk_duration_ms = end_ms - start_ms
            
            # 너무 짧은 발화: 다음과 병합 (3초 미만)
            if chunk_duration_ms < 3000 and i < len(nonsilent_ranges) - 1:
                # 병합할 청크들 수집
                merged_start = start_ms
                merged_end = end_ms
                merged_indices = [i]
                
                j = i + 1
                while j < len(nonsilent_ranges):
                    next_start, next_end = nonsilent_ranges[j]
                    
                    # 다음 발화까지의 간격 확인
                    gap_ms = next_start - merged_end
                    
                    # 간격이 2초 이내이고 총 길이가 20초 미만이면 병합
                    if gap_ms < 2000 and (next_end - merged_start) < 20000:
                        merged_end = next_end
                        merged_indices.append(j)
                        j += 1
                        
                        # 충분한 길이가 되면 중단
                        if merged_end - merged_start >= 8000:  # 8초 이상
                            break
                    else:
                        break
                
                # 병합된 청크 생성 (실제 위치 보존)
                optimized_chunks.append({
                    'start_ms': merged_start,  # 실제 시작 위치
                    'end_ms': merged_end,      # 실제 끝 위치
                    'is_merged': len(merged_indices) > 1,
                    'merged_indices': merged_indices if len(merged_indices) > 1 else None,
                    'original_ranges': [nonsilent_ranges[idx] for idx in merged_indices]
                })
                i = j
                
            # 너무 긴 발화: 분할 (20초 초과)
            elif chunk_duration_ms > 20000:
                # 15초 단위로 분할
                chunk_start = start_ms
                while chunk_start < end_ms:
                    chunk_end = min(chunk_start + 15000, end_ms)
                    
                    optimized_chunks.append({
                        'start_ms': chunk_start,  # 실제 시작 위치
                        'end_ms': chunk_end,      # 실제 끝 위치
                        'is_merged': False,
                        'merged_indices': None,
                        'original_ranges': [(chunk_start, chunk_end)]
                    })
                    
                    chunk_start = chunk_end
                i += 1
                
            # 적절한 길이: 그대로 사용
            else:
                optimized_chunks.append({
                    'start_ms': start_ms,  # 실제 시작 위치
                    'end_ms': end_ms,      # 실제 끝 위치
                    'is_merged': False,
                    'merged_indices': None,
                    'original_ranges': [(start_ms, end_ms)]
                })
                i += 1
        
        # ChunkInfo 객체 생성 및 오디오 저장
        final_chunks = []
        for idx, chunk_data in enumerate(optimized_chunks):
            # 실제 위치에서 오디오 추출 (앞뒤 여유 포함)
            extract_start = max(0, chunk_data['start_ms'] - 200)  # 0.2초 앞
            extract_end = min(duration_ms, chunk_data['end_ms'] + 200)  # 0.2초 뒤
            
            chunk_audio = audio[extract_start:extract_end]
            
            # 청크 저장
            chunk_path = os.path.join(temp_dir, f"sentence_{idx:04d}.wav")
            chunk_audio.export(chunk_path, format="wav")
            
            chunk_info = ChunkInfo(
                index=idx,
                path=chunk_path,
                start_ms=chunk_data['start_ms'],  # 실제 시작 시간
                end_ms=chunk_data['end_ms'],      # 실제 끝 시간
                duration_ms=chunk_data['end_ms'] - chunk_data['start_ms'],
                has_speech=True,
                overlap_with_next=0,
                is_merged=chunk_data.get('is_merged', False),
                merged_indices=chunk_data.get('merged_indices')
            )
            final_chunks.append(chunk_info)
            
            print(f"  청크 {idx + 1}: {chunk_data['start_ms']/1000:.1f}초 ~ {chunk_data['end_ms']/1000:.1f}초 (실제 위치)")
        
        print(f"📦 문장 청크: {len(nonsilent_ranges)}개 발화 → {len(final_chunks)}개 청크로 최적화")
        return final_chunks
    
    def _create_smart_chunks(self, audio_path: str) -> List[ChunkInfo]:
        """오디오 파일을 스마트하게 분할 (기존 방식)"""
        if not PYDUB_AVAILABLE:
            return []
        
        from pydub.silence import detect_nonsilent
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        
        # 무음이 아닌 구간 감지
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=300,  # 0.3초 이상 무음
            silence_thresh=-40,    # dB
            seek_step=10
        )
        
        chunks = []
        temp_dir = tempfile.mkdtemp()
        chunk_index = 0
        
        # 최대 15초, 최소 5초 청크
        max_chunk_ms = 15000
        min_chunk_ms = 5000
        target_overlap_ms = 3000  # 3초 오버랩
        
        current_start = 0
        
        print(f"📊 스마트 분할: 총 {duration_ms/1000:.1f}초")
        
        while current_start < duration_ms:
            # 이상적인 끝 지점
            ideal_end = min(current_start + max_chunk_ms, duration_ms)
            
            # 실제 끝 지점 찾기 (무음 구간 찾기)
            actual_end = ideal_end
            
            # ideal_end 근처의 무음 구간 찾기
            for i in range(len(nonsilent_ranges) - 1):
                silence_start = nonsilent_ranges[i][1]
                silence_end = nonsilent_ranges[i + 1][0]
                
                # 무음 구간이 ideal_end 근처에 있으면
                if abs(silence_start - ideal_end) < 2000:  # 2초 이내
                    actual_end = silence_start
                    break
            
            # 청크가 너무 짧으면 조정
            if actual_end - current_start < min_chunk_ms and actual_end < duration_ms:
                actual_end = min(current_start + max_chunk_ms, duration_ms)
            
            # 청크 추출 (오버랩 포함)
            chunk_end = min(actual_end + target_overlap_ms, duration_ms)
            chunk = audio[current_start:chunk_end]
            
            # 저장
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(ChunkInfo(
                index=chunk_index,
                path=chunk_path,
                start_ms=current_start,
                end_ms=chunk_end,
                duration_ms=chunk_end - current_start,
                has_speech=True,
                overlap_with_next=target_overlap_ms if chunk_end < duration_ms else 0
            ))
            
            print(f"  청크 {chunk_index + 1}: {current_start/1000:.1f}초 ~ {chunk_end/1000:.1f}초 ({(chunk_end-current_start)/1000:.1f}초)")
            
            # 다음 청크 시작점 (오버랩 적용)
            current_start = actual_end - target_overlap_ms
            if current_start >= duration_ms - min_chunk_ms:
                break
            
            chunk_index += 1
        
        print(f"📊 총 {len(chunks)}개 청크 생성됨")
        return chunks
    
    def _api_call_with_retry(self, chunk_path: str, language: str, 
                           max_retries: int = 2) -> ProcessResult:
        """API 호출 with 재시도 로직"""
        last_error = None
        
        for retry in range(max_retries + 1):
            try:
                # 파일 읽기
                with open(chunk_path, "rb") as f:
                    audio_contents = base64.b64encode(f.read()).decode("utf8")
                
                # API 요청
                request_json = {
                    "argument": {
                        "language_code": language,
                        "audio": audio_contents
                    }
                }
                
                start_time = time.time()
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
                duration = time.time() - start_time
                
                if response.status == 200:
                    result = json.loads(response.data.decode("utf-8"))
                    if result.get("result", -1) == 0:
                        text = result.get("return_object", {}).get("recognized", "")
                        return ProcessResult(
                            success=True,
                            text=text,
                            chunk_index=0,
                            duration=duration,
                            retry_count=retry
                        )
                    else:
                        last_error = f"API Error: {result.get('reason', 'Unknown')}"
                else:
                    last_error = f"HTTP Error: {response.status}"
                
                # 재시도 전 대기 (exponential backoff)
                if retry < max_retries:
                    wait_time = (2 ** retry) * 1.0  # 1초, 2초, 4초...
                    time.sleep(wait_time)
                    self.processing_stats['retry_count'] += 1
                    
            except Exception as e:
                last_error = str(e)
                if retry < max_retries:
                    time.sleep(1.0)
        
        return ProcessResult(
            success=False,
            text="",
            chunk_index=0,
            duration=0,
            retry_count=max_retries,
            error=last_error
        )
    
    def _process_single(self, chunk: ChunkInfo, language: str, 
                       progress_callback: Optional[Callable]) -> List[ProcessResult]:
        """단일 청크 처리"""
        if progress_callback:
            progress_callback(50, 100, "음성 인식 중...", None)
        
        result = self._api_call_with_retry(chunk.path, language)
        result.chunk_index = chunk.index
        
        # 타임스탬프 정보 추가
        result.start_time = chunk.start_ms / 1000.0
        result.end_time = chunk.end_ms / 1000.0
        
        if result.success:
            self.processing_stats['success_chunks'] = 1
        else:
            self.processing_stats['failed_chunks'] = 1
        
        if progress_callback:
            progress_callback(100, 100, "완료!", None)
        
        return [result]
    
    def _process_sequential(self, chunks: List[ChunkInfo], language: str,
                          progress_callback: Optional[Callable]) -> List[ProcessResult]:
        """순차 처리"""
        results = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # 진행률 계산
            percent = int((i / total) * 100)
            eta = self._calculate_eta(i, total, results)
            
            if progress_callback:
                progress_callback(percent, 100, f"청크 {i+1}/{total} 처리 중...", eta)
            
            # 음성이 없는 청크는 건너뛰기
            if not chunk.has_speech:
                results.append(ProcessResult(
                    success=True,
                    text="",
                    chunk_index=chunk.index,
                    duration=0,
                    start_time=chunk.start_ms / 1000.0,
                    end_time=chunk.end_ms / 1000.0
                ))
                continue
            
            # API 호출
            result = self._api_call_with_retry(chunk.path, language)
            result.chunk_index = chunk.index
            
            # 타임스탬프 정보 추가
            result.start_time = chunk.start_ms / 1000.0
            result.end_time = chunk.end_ms / 1000.0
            
            results.append(result)
            
            # 통계 업데이트
            if result.success:
                self.processing_stats['success_chunks'] += 1
            else:
                self.processing_stats['failed_chunks'] += 1
            
            # 다음 요청 전 대기
            if i < total - 1:
                time.sleep(1.0)
        
        return results
    
    def _process_parallel(self, chunks: List[ChunkInfo], language: str,
                        max_workers: int, progress_callback: Optional[Callable]) -> List[ProcessResult]:
        """병렬 처리 (최대 2워커)"""
        results = []
        total = len(chunks)
        completed = 0
        
        # 워커 수 제한
        max_workers = min(max_workers, 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_chunk = {}
            
            for i, chunk in enumerate(chunks):
                # 음성 없는 청크는 즉시 처리
                if not chunk.has_speech:
                    results.append(ProcessResult(
                        success=True,
                        text="",
                        chunk_index=chunk.index,
                        duration=0,
                        start_time=chunk.start_ms / 1000.0,
                        end_time=chunk.end_ms / 1000.0
                    ))
                    completed += 1
                    continue
                
                # 2개씩 배치로 제출 (API 부하 관리)
                if i > 0 and i % 2 == 0:
                    time.sleep(0.5)
                
                future = executor.submit(
                    self._api_call_with_retry,
                    chunk.path,
                    language
                )
                future_to_chunk[future] = chunk
            
            # 결과 수집
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                
                try:
                    result = future.result(timeout=30)
                    result.chunk_index = chunk.index
                    
                    # 타임스탬프 정보 추가
                    result.start_time = chunk.start_ms / 1000.0
                    result.end_time = chunk.end_ms / 1000.0
                    
                    results.append(result)
                    
                    if result.success:
                        self.processing_stats['success_chunks'] += 1
                    else:
                        self.processing_stats['failed_chunks'] += 1
                        # 실패한 청크 재시도 (한 번 더)
                        if result.retry_count < 3:
                            time.sleep(2.0)
                            retry_result = self._api_call_with_retry(chunk.path, language, max_retries=1)
                            if retry_result.success:
                                results[-1] = retry_result  # 교체
                                self.processing_stats['success_chunks'] += 1
                                self.processing_stats['failed_chunks'] -= 1
                    
                except Exception as e:
                    results.append(ProcessResult(
                        success=False,
                        text="",
                        chunk_index=chunk.index,
                        duration=0,
                        error=str(e),
                        start_time=chunk.start_ms / 1000.0,
                        end_time=chunk.end_ms / 1000.0
                    ))
                    self.processing_stats['failed_chunks'] += 1
                
                completed += 1
                
                # 진행률 업데이트
                if progress_callback:
                    percent = int((completed / total) * 100)
                    eta = self._calculate_eta(completed, total, results)
                    progress_callback(percent, 100, f"병렬 처리 중... {completed}/{total}", eta)
        
        # 인덱스 순으로 정렬
        results.sort(key=lambda x: x.chunk_index)
        return results
    
    def _organize_with_timestamps(self, results: List[ProcessResult], 
                                  chunks: List[ChunkInfo]) -> List[Dict]:
        """타임스탬프와 함께 결과 정리 (개선된 버전)"""
        sentences = []
        
        for result in results:
            if not result.success or not result.text:
                continue
            
            # 해당 청크 찾기
            chunk = next((c for c in chunks if c.index == result.chunk_index), None)
            if not chunk:
                continue
            
            # 이미 정확한 시간이 있음
            actual_start = result.start_time  # 이미 초 단위
            actual_end = result.end_time
            
            if chunk.is_merged and chunk.merged_indices:
                # 병합된 청크: 문장 분리 후 시간 배분
                text_sentences = self._split_text_to_sentences(result.text)
                
                if len(text_sentences) == 1:
                    # 단일 문장
                    sentences.append({
                        'text': result.text.strip(),
                        'start_time': actual_start,
                        'end_time': actual_end
                    })
                else:
                    # 여러 문장: 글자 수 비례로 시간 배분 (균등 분배보다 정확)
                    total_chars = sum(len(s) for s in text_sentences)
                    total_duration = actual_end - actual_start
                    current_time = actual_start
                    
                    for sent_text in text_sentences:
                        # 글자 수 비례로 시간 할당
                        if total_chars > 0:
                            sent_duration = (len(sent_text) / total_chars) * total_duration
                        else:
                            sent_duration = total_duration / len(text_sentences)
                        
                        sentences.append({
                            'text': sent_text.strip(),
                            'start_time': current_time,
                            'end_time': min(current_time + sent_duration, actual_end)
                        })
                        current_time += sent_duration
            else:
                # 단일 청크
                sentences.append({
                    'text': result.text.strip(),
                    'start_time': actual_start,
                    'end_time': actual_end
                })
        
        # 시간순 정렬 (이미 정렬되어 있겠지만 확실히)
        sentences.sort(key=lambda x: x['start_time'])
        
        # 중복 제거 및 시간 조정
        if len(sentences) > 1:
            cleaned_sentences = []
            prev_sentence = None
            
            for sentence in sentences:
                # 이전 문장과 겹치는 경우 조정
                if prev_sentence and sentence['start_time'] < prev_sentence['end_time']:
                    # 시작 시간 조정
                    sentence['start_time'] = prev_sentence['end_time']
                
                # 시작이 끝보다 늦은 경우 스킵
                if sentence['start_time'] >= sentence['end_time']:
                    continue
                
                cleaned_sentences.append(sentence)
                prev_sentence = sentence
            
            sentences = cleaned_sentences
        
        return sentences
    
    def _split_text_to_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        import re
        
        # 문장 종결 패턴으로 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences if sentences else [text]
    
    def _extract_plain_text(self, sentences: List[Dict]) -> str:
        """일반 텍스트 추출"""
        return " ".join(s['text'] for s in sentences if s.get('text'))
    
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
    
    def _merge_results(self, results: List[ProcessResult]) -> str:
        """결과 병합 및 중복 제거 (기존 방식)"""
        if not results:
            return ""
        
        texts = []
        previous_text = ""
        
        for i, result in enumerate(results):
            if not result.success or not result.text:
                # 실패한 청크 처리 - 이전/이후 청크의 오버랩 부분 활용
                if i > 0 and i < len(results) - 1:
                    # 이전과 다음 청크 사이를 보간
                    texts.append("[...]")  # 누락 표시
                continue
            
            current_text = result.text.strip()
            
            # 오버랩 중복 제거
            if i > 0 and previous_text:
                # 마지막 n글자와 첫 n글자 비교
                overlap_found = False
                for check_len in range(min(50, len(previous_text)//2), 5, -1):
                    last_part = previous_text[-check_len:]
                    if current_text.startswith(last_part):
                        current_text = current_text[check_len:].strip()
                        overlap_found = True
                        break
                
                # 단어 단위 중복 체크 (더 정교하게)
                if not overlap_found:
                    prev_words = previous_text.split()[-5:]  # 마지막 5단어
                    curr_words = current_text.split()[:5]    # 처음 5단어
                    
                    for j in range(min(len(prev_words), len(curr_words))):
                        if prev_words[-j-1:] == curr_words[:j+1]:
                            current_text = ' '.join(curr_words[j+1:] + current_text.split()[5:])
                            break
            
            if current_text:
                # 문장 부호 정리
                if texts and not texts[-1].endswith(('.', '!', '?')):
                    if not current_text[0].isupper():
                        texts[-1] += '.'
                
                texts.append(current_text)
                previous_text = result.text.strip()
        
        # 최종 결합
        final_text = ' '.join(texts)
        
        # 후처리
        final_text = ' '.join(final_text.split())  # 중복 공백 제거
        final_text = final_text.replace(' .', '.')
        final_text = final_text.replace(' ,', ',')
        final_text = final_text.replace(' ?', '?')
        final_text = final_text.replace(' !', '!')
        
        return final_text
    
    def _calculate_eta(self, completed: int, total: int, 
                      results: List[ProcessResult]) -> Optional[int]:
        """예상 완료 시간 계산"""
        if completed == 0 or not results:
            return None
        
        # 평균 처리 시간 계산
        avg_duration = sum(r.duration for r in results if r.duration > 0) / max(1, len(results))
        remaining = total - completed
        eta_seconds = int(remaining * avg_duration)
        
        return eta_seconds


# 간편 사용 함수
def transcribe_audio(audio_path: str, api_key: str, 
                     max_workers: int = 2,
                     progress_callback: Optional[Callable] = None,
                     enable_timestamps: bool = True) -> Dict:
    """
    음성을 텍스트로 변환 (타임스탬프 옵션 추가)
    
    Args:
        audio_path: 오디오 파일 경로
        api_key: ETRI API 키
        max_workers: 병렬 워커 수 (1-2)
        progress_callback: 진행 콜백
        enable_timestamps: 타임스탬프 포함 여부
    
    Returns:
        {
            'success': bool, 
            'text': str, 
            'formatted_text': str (타임스탬프 포함),
            'sentences': list (문장별 정보),
            'stats': dict
        }
    """
    processor = ImprovedSTTProcessor(api_key)
    return processor.process(
        audio_path, 
        max_workers=max_workers, 
        progress_callback=progress_callback,
        enable_timestamps=enable_timestamps
    )