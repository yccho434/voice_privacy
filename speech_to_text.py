# speech_to_text.py
"""
ETRI 음성인식 API 모듈
긴 음성 파일 처리를 위한 청크 분할 + 병렬 처리 지원
"""

import json
import base64
import urllib3
import time
import tempfile
import os
import sys
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# FFmpeg 자동 설정
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # pydub에 FFmpeg 경로 설정
    from pydub.utils import which
    from pydub import AudioSegment
    
    if not which("ffmpeg"):
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
        print(f"✅ FFmpeg 자동 설정 완료: {ffmpeg_path}")
    
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub 또는 imageio-ffmpeg 설치 필요")

# Windows Chocolatey FFmpeg 경로 추가 (폴백)
if sys.platform == "win32" and PYDUB_AVAILABLE:
    choco_path = r"C:\ProgramData\chocolatey\bin"
    if os.path.exists(os.path.join(choco_path, "ffmpeg.exe")):
        os.environ["PATH"] = choco_path + os.pathsep + os.environ.get("PATH", "")


@dataclass
class ChunkResult:
    """청크 처리 결과"""
    chunk_index: int
    text: str
    success: bool
    error: Optional[str] = None
    duration: float = 0.0


class ETRISpeechToText:
    """ETRI 음성인식 API 클라이언트 (병렬 처리 지원)"""
    
    API_URL = "http://epretx.etri.re.kr:8000/api/WiseASR_Recognition"
    MAX_CHUNK_SECONDS = 15  # 청크 크기
    OVERLAP_SECONDS = 3.0    # 오버랩
    REQUEST_DELAY = 1.0      # 순차 처리시 요청 간 딜레이
    PARALLEL_BATCH_DELAY = 0.3  # 병렬 처리시 배치 간 딜레이
    
    def __init__(self, access_key: str):
        """
        Args:
            access_key: ETRI API 접근 키
        """
        self.access_key = access_key
        self.http = urllib3.PoolManager()
    
    def recognize(self, audio_file_path: str, language: str = "korean") -> Dict:
        """
        음성 파일을 텍스트로 변환 (기본 메서드)
        
        Args:
            audio_file_path: 오디오 파일 경로
            language: 언어 코드 (korean, english, japanese, chinese 등)
            
        Returns:
            {"success": bool, "text": str, "error": str}
        """
        try:
            # 파일 읽기 및 Base64 인코딩
            with open(audio_file_path, "rb") as f:
                audio_contents = base64.b64encode(f.read()).decode("utf8")
            
            # API 요청 구성
            request_json = {
                "argument": {
                    "language_code": language,
                    "audio": audio_contents
                }
            }
            
            # API 호출
            response = self.http.request(
                "POST",
                self.API_URL,
                headers={
                    "Content-Type": "application/json; charset=UTF-8",
                    "Authorization": self.access_key
                },
                body=json.dumps(request_json)
            )
            
            if response.status == 200:
                result = json.loads(response.data.decode("utf-8"))
                
                if result.get("result", -1) == 0:
                    # 인식된 텍스트 추출
                    return_obj = result.get("return_object", {})
                    recognized_text = return_obj.get("recognized", "")
                    
                    # recognized 필드가 없으면 다른 필드 확인
                    if not recognized_text:
                        recognized_text = return_obj.get("result", "")
                        if not recognized_text:
                            recognized_text = return_obj.get("text", "")
                    
                    return {
                        "success": True,
                        "text": recognized_text,
                        "error": None
                    }
                else:
                    error_msg = f"API Error Code: {result.get('result')}"
                    if result.get('reason'):
                        error_msg += f" - Reason: {result.get('reason')}"
                    
                    return {
                        "success": False,
                        "text": "",
                        "error": error_msg
                    }
            else:
                return {
                    "success": False,
                    "text": "",
                    "error": f"HTTP Error {response.status}"
                }
                
        except FileNotFoundError:
            return {
                "success": False,
                "text": "",
                "error": f"파일을 찾을 수 없습니다: {audio_file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": f"오류 발생: {str(e)}"
            }
    
    def split_audio_smart(self, audio_path: str) -> List[str]:
        """
        오디오 파일을 스마트하게 분할 (무음 구간 활용)
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            청크 파일 경로 리스트
        """
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub가 필요합니다. pip install pydub")
        
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
            chunks.append(chunk_path)
            
            print(f"  청크 {chunk_index + 1}: {current_start/1000:.1f}초 ~ {chunk_end/1000:.1f}초 ({(chunk_end-current_start)/1000:.1f}초)")
            
            # 다음 청크 시작점 (오버랩 적용)
            current_start = actual_end - target_overlap_ms
            if current_start >= duration_ms - min_chunk_ms:
                break
            
            chunk_index += 1
        
        print(f"📊 총 {len(chunks)}개 청크 생성됨")
        return chunks
    
    def split_audio(self, audio_path: str, chunk_seconds: int = None) -> List[str]:
        """
        오디오 파일을 청크로 분할 (단순 분할, 폴백용)
        
        Args:
            audio_path: 오디오 파일 경로
            chunk_seconds: 청크 크기 (초)
            
        Returns:
            청크 파일 경로 리스트
        """
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub가 필요합니다. pip install pydub")
        
        chunk_seconds = chunk_seconds or self.MAX_CHUNK_SECONDS
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunk_ms = chunk_seconds * 1000
        overlap_ms = int(self.OVERLAP_SECONDS * 1000)  # 3초 오버랩
        
        chunks = []
        temp_dir = tempfile.mkdtemp()
        
        # 청크 생성
        start_ms = 0
        chunk_index = 0
        
        print(f"📊 단순 분할: 총 {duration_ms/1000:.1f}초, {chunk_seconds}초 청크, {self.OVERLAP_SECONDS}초 오버랩")
        
        while start_ms < duration_ms:
            # 청크 추출 (오버랩 포함하여 더 길게)
            end_ms = min(start_ms + chunk_ms + overlap_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            # 임시 파일로 저장
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            
            print(f"  청크 {chunk_index + 1}: {start_ms/1000:.1f}초 ~ {end_ms/1000:.1f}초")
            
            # 다음 청크 시작점 (오버랩 없이)
            start_ms = start_ms + chunk_ms
            if start_ms >= duration_ms:
                break
                
            chunk_index += 1
        
        print(f"📊 총 {len(chunks)}개 청크 생성됨")
        return chunks
    
    def _process_sequential(self, chunk_paths: List[str], language: str, 
                          progress_callback: Optional[Callable] = None) -> Tuple[List[ChunkResult], List[str]]:
        """
        순차 처리 (기존 방식)
        
        Returns:
            (청크 결과 리스트, 텍스트 리스트)
        """
        chunks_results = []
        all_texts = []
        previous_text = ""
        
        for i, chunk_path in enumerate(chunk_paths):
            # 진행 상황 알림
            if progress_callback:
                percent = int((i / len(chunk_paths)) * 100)
                progress_callback(percent, 100, f"청크 {i+1}/{len(chunk_paths)} 처리 중...")
            
            # API 호출
            start_time = time.time()
            result = self.recognize(chunk_path, language)
            duration = time.time() - start_time
            
            # 결과 처리 (중복 제거)
            current_text = result.get("text", "").strip()
            
            # 오버랩 중복 제거
            if i > 0 and previous_text and current_text:
                max_overlap = min(100, len(previous_text) // 2, len(current_text) // 2)
                
                for check_len in range(max_overlap, 10, -1):
                    last_part = previous_text[-check_len:]
                    idx = current_text.find(last_part)
                    if idx != -1 and idx < 50:
                        current_text = current_text[idx + len(last_part):].strip()
                        break
            
            if current_text:
                chunk_result = ChunkResult(
                    chunk_index=i,
                    text=current_text,
                    success=result["success"],
                    error=result.get("error"),
                    duration=duration
                )
                chunks_results.append(chunk_result)
                all_texts.append(current_text)
                previous_text = result.get("text", "").strip()
            
            # 다음 요청 전 딜레이
            if i < len(chunk_paths) - 1:
                time.sleep(self.REQUEST_DELAY)
        
        return chunks_results, all_texts
    
    def _process_parallel(self, chunk_paths: List[str], language: str, max_workers: int,
                         progress_callback: Optional[Callable] = None) -> Tuple[List[ChunkResult], List[str]]:
        """
        병렬 처리 (새로운 기능)
        
        Args:
            max_workers: 동시 처리 워커 수 (1 또는 2)
            
        Returns:
            (청크 결과 리스트, 텍스트 리스트)
        """
        chunks_results = []
        all_texts = []
        total_chunks = len(chunk_paths)
        
        print(f"⚡ 병렬 처리 모드 ({max_workers} 워커) - {total_chunks}개 청크")
        
        # 첫 청크는 테스트용 순차 처리
        if progress_callback:
            progress_callback(0, total_chunks, "연결 테스트 중...")
        
        first_result = self.recognize(chunk_paths[0], language)
        if not first_result["success"]:
            print("⚠️ 첫 청크 실패, 순차 모드로 전환")
            return self._process_sequential(chunk_paths, language, progress_callback)
        
        chunks_results.append(ChunkResult(0, first_result.get("text", ""), True))
        all_texts.append(first_result.get("text", ""))
        
        # 나머지 청크 병렬 처리
        remaining_chunks = chunk_paths[1:]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # 작업 제출
            for i, chunk_path in enumerate(remaining_chunks, 1):
                # 워커 수에 따른 딜레이 조정
                if max_workers == 2 and i % 2 == 0:
                    time.sleep(self.PARALLEL_BATCH_DELAY)  # 2개마다 딜레이
                
                future = executor.submit(self.recognize, chunk_path, language)
                futures.append((i, future, chunk_path))
                
                if progress_callback:
                    progress_callback(i, total_chunks, f"청크 {i+1}/{total_chunks} 제출 중...")
            
            # 결과 수집
            print("⏳ 결과 대기 중...")
            for chunk_index, future, chunk_path in futures:
                try:
                    result = future.result(timeout=30)
                    
                    if result["success"]:
                        chunks_results.append(ChunkResult(
                            chunk_index,
                            result.get("text", ""),
                            True
                        ))
                        all_texts.append(result.get("text", ""))
                        print(f"  ✅ 청크 {chunk_index+1} 완료")
                    else:
                        # 실패 시 재시도
                        print(f"  ⚠️ 청크 {chunk_index+1} 실패, 재시도...")
                        time.sleep(2)
                        retry_result = self.recognize(chunk_path, language)
                        
                        if retry_result["success"]:
                            all_texts.append(retry_result.get("text", ""))
                            print(f"  ✅ 청크 {chunk_index+1} 재시도 성공")
                        else:
                            chunks_results.append(ChunkResult(
                                chunk_index,
                                "",
                                False,
                                retry_result.get("error")
                            ))
                            print(f"  ❌ 청크 {chunk_index+1} 최종 실패")
                    
                except Exception as e:
                    print(f"❌ 청크 {chunk_index+1} 예외: {e}")
                    chunks_results.append(ChunkResult(
                        chunk_index,
                        "",
                        False,
                        str(e)
                    ))
                
                if progress_callback:
                    percent = int((chunk_index / total_chunks) * 100)
                    progress_callback(percent, 100, f"청크 {chunk_index+1}/{total_chunks} 처리 완료")
        
        return chunks_results, all_texts
    
    def recognize_long_audio(
        self, 
        audio_path: str, 
        language: str = "korean",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        debug_mode: bool = False,
        max_workers: int = 1  # 병렬 처리 워커 수 (1=순차, 2=병렬)
    ) -> Dict:
        """
        긴 오디오 파일 처리 (병렬 처리 옵션 포함)
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드
            progress_callback: 진행 상황 콜백 (current, total, status)
            debug_mode: 디버그 모드
            max_workers: 병렬 처리 워커 수 (1=순차, 2=병렬)
            
        Returns:
            {"success": bool, "text": str, "chunks": List[ChunkResult], "error": str, "processing_mode": str}
        """
        if not PYDUB_AVAILABLE:
            # pydub 없으면 기본 메서드로 폴백
            result = self.recognize(audio_path, language)
            return {
                **result,
                "chunks": [ChunkResult(0, result.get("text", ""), result["success"], result.get("error"))],
                "processing_mode": "single"
            }
        
        try:
            start_total_time = time.time()
            
            # 오디오 파일 정보 확인
            audio = AudioSegment.from_file(audio_path)
            duration_seconds = len(audio) / 1000
            
            print(f"📊 오디오 정보: {duration_seconds:.1f}초, {len(audio)}ms")
            
            # 20초 이하면 바로 처리
            if duration_seconds <= self.MAX_CHUNK_SECONDS:
                result = self.recognize(audio_path, language)
                return {
                    **result,
                    "chunks": [ChunkResult(0, result.get("text", ""), result["success"], result.get("error"))],
                    "duration_seconds": duration_seconds,
                    "processing_mode": "single"
                }
            
            # 청크 분할
            if progress_callback:
                progress_callback(0, 100, "오디오 파일 분할 중...")
            
            try:
                chunk_paths = self.split_audio_smart(audio_path)
                print("✅ 스마트 분할 사용")
            except Exception as e:
                print(f"⚠️ 스마트 분할 실패, 단순 분할 사용: {e}")
                chunk_paths = self.split_audio(audio_path)
            
            total_chunks = len(chunk_paths)
            
            # 처리 모드 결정
            if max_workers == 1 or total_chunks <= 3:
                # 순차 처리
                processing_mode = "sequential"
                chunks_results, all_texts = self._process_sequential(
                    chunk_paths, language, progress_callback
                )
            else:
                # 병렬 처리
                processing_mode = f"parallel_{max_workers}_workers"
                chunks_results, all_texts = self._process_parallel(
                    chunk_paths, language, max_workers, progress_callback
                )
            
            # 임시 파일 정리
            for chunk_path in chunk_paths:
                try:
                    os.remove(chunk_path)
                except:
                    pass
            
            # 완료 알림
            if progress_callback:
                progress_callback(100, 100, "완료!")
            
            # 전체 텍스트 결합
            if all_texts:
                processed_texts = []
                for text in all_texts:
                    text = text.strip()
                    if text and not text[-1] in '.!?':
                        text += '.'
                    processed_texts.append(text)
                
                combined_text = ' '.join(processed_texts)
                combined_text = ' '.join(combined_text.split())
            else:
                combined_text = ""
            
            # 성공 여부 판단
            success_count = sum(1 for c in chunks_results if c.success)
            total_time = time.time() - start_total_time
            
            return {
                "success": success_count > 0,
                "text": combined_text,
                "chunks": chunks_results,
                "total_chunks": total_chunks,
                "success_chunks": success_count,
                "duration_seconds": duration_seconds,
                "processing_time": total_time,
                "processing_mode": processing_mode,
                "error": None if success_count == total_chunks else f"{total_chunks - success_count}개 청크 실패"
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "chunks": [],
                "error": f"처리 중 오류: {str(e)}",
                "processing_mode": "error"
            }


# 간단한 테스트
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python speech_to_text.py [오디오파일] [워커수(1또는2)]")
        print("예시: python speech_to_text.py test.mp3 2")
        sys.exit(1)
    
    api_key = os.getenv("ETRI_API_KEY", "YOUR_API_KEY")
    audio_file = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    stt = ETRISpeechToText(api_key)
    result = stt.recognize_long_audio(audio_file, max_workers=max_workers)
    
    if result["success"]:
        print(f"\n✅ 인식 성공!")
        print(f"처리 모드: {result.get('processing_mode')}")
        print(f"처리 시간: {result.get('processing_time', 0):.1f}초")
        print(f"텍스트 길이: {len(result['text'])}자")
        print(f"\n인식 결과:\n{result['text'][:500]}...")  # 처음 500자만 출력
    else:
        print(f"❌ 오류: {result['error']}")