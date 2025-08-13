# speech_to_text.py
"""
ETRI 음성인식 API 모듈
긴 음성 파일 처리를 위한 청크 분할 기능 포함
"""

import json
import base64
import urllib3
import time
import tempfile
import os
import sys
from typing import Optional, Dict, List, Callable
from pathlib import Path
from dataclasses import dataclass

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
    """ETRI 음성인식 API 클라이언트"""
    
    API_URL = "http://epretx.etri.re.kr:8000/api/WiseASR_Recognition"
    MAX_CHUNK_SECONDS = 15  # 청크 크기를 15초로 줄임 (더 안전)
    OVERLAP_SECONDS = 3.0    # 오버랩을 3초로 대폭 증가
    REQUEST_DELAY = 1.0      # 요청 간 딜레이 (초)
    
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
            print(f"🔍 API 호출 중... 언어: {language}, 파일: {audio_file_path}")
            response = self.http.request(
                "POST",
                self.API_URL,
                headers={
                    "Content-Type": "application/json; charset=UTF-8",
                    "Authorization": self.access_key
                },
                body=json.dumps(request_json)
            )
            
            # 응답 처리
            print(f"📡 응답 상태 코드: {response.status}")
            
            if response.status == 200:
                result = json.loads(response.data.decode("utf-8"))
                
                # 디버그: 전체 응답 구조 출력
                print(f"📋 응답 result 코드: {result.get('result')}")
                
                # 성공 체크
                if result.get("result", -1) == 0:
                    # 인식된 텍스트 추출
                    return_obj = result.get("return_object", {})
                    recognized_text = return_obj.get("recognized", "")
                    
                    # recognized 필드가 없으면 다른 필드 확인
                    if not recognized_text:
                        # 다른 가능한 필드들 확인
                        recognized_text = return_obj.get("result", "")
                        if not recognized_text:
                            recognized_text = return_obj.get("text", "")
                        if not recognized_text:
                            print(f"⚠️ 텍스트 필드를 찾을 수 없음. return_object 구조: {list(return_obj.keys())}")
                    
                    print(f"✅ 인식 성공: {len(recognized_text)}자")
                    
                    return {
                        "success": True,
                        "text": recognized_text,
                        "error": None
                    }
                else:
                    # 오류 상세 정보
                    error_msg = f"API Error Code: {result.get('result')}"
                    if result.get('reason'):
                        error_msg += f" - Reason: {result.get('reason')}"
                    if result.get('message'):
                        error_msg += f" - Message: {result.get('message')}"
                    
                    print(f"❌ API 오류: {error_msg}")
                    print(f"📋 전체 응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    
                    return {
                        "success": False,
                        "text": "",
                        "error": error_msg
                    }
            else:
                # HTTP 오류 상세 정보
                error_body = response.data.decode("utf-8")
                error_msg = f"HTTP Error {response.status}"
                
                try:
                    error_json = json.loads(error_body)
                    if error_json.get('message'):
                        error_msg += f": {error_json.get('message')}"
                    if error_json.get('error'):
                        error_msg += f" - {error_json.get('error')}"
                except:
                    error_msg += f": {error_body[:200]}"  # 처음 200자만
                
                print(f"❌ HTTP 오류: {error_msg}")
                
                return {
                    "success": False,
                    "text": "",
                    "error": error_msg
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
    
    def recognize_long_audio(
        self, 
        audio_path: str, 
        language: str = "korean",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        debug_mode: bool = False  # debug_mode 매개변수 추가
    ) -> Dict:
        """
        긴 오디오 파일 처리 (자동 분할, 오버랩 중복 제거)
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드
            progress_callback: 진행 상황 콜백 (current, total, status)
            
        Returns:
            {"success": bool, "text": str, "chunks": List[ChunkResult], "error": str}
        """
        if not PYDUB_AVAILABLE:
            # pydub 없으면 기본 메서드로 폴백 (20초 이하만 가능)
            result = self.recognize(audio_path, language)
            return {
                **result,
                "chunks": [ChunkResult(0, result.get("text", ""), result["success"], result.get("error"))]
            }
        
        try:
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
                    "duration_seconds": duration_seconds
                }
            
            # 청크 분할 (스마트 분할 시도, 실패시 단순 분할)
            if progress_callback:
                progress_callback(0, 100, "오디오 파일 분할 중...")
            
            try:
                # 무음 구간 기반 스마트 분할 시도
                chunk_paths = self.split_audio_smart(audio_path)
                print("✅ 스마트 분할 사용")
            except Exception as e:
                print(f"⚠️ 스마트 분할 실패, 단순 분할 사용: {e}")
                # 폴백: 단순 분할
                chunk_paths = self.split_audio(audio_path)
            
            total_chunks = len(chunk_paths)
            
            # 예상 시간 계산 (더 정확하게)
            estimated_time = total_chunks * (2.5 + self.REQUEST_DELAY)  # API 응답 시간 포함
            print(f"⏱️ 예상 처리 시간: {estimated_time:.0f}초 ({total_chunks}개 청크)")
            
            # 청크별 처리
            chunks_results: List[ChunkResult] = []
            all_texts = []
            previous_text = ""  # 이전 청크의 마지막 부분 저장 (중복 제거용)
            
            for i, chunk_path in enumerate(chunk_paths):
                # 진행 상황 알림
                if progress_callback:
                    percent = int((i / total_chunks) * 100)
                    remaining_time = (total_chunks - i) * (2.5 + self.REQUEST_DELAY)
                    progress_callback(percent, 100, f"청크 {i+1}/{total_chunks} 처리 중... (남은 시간: 약 {int(remaining_time)}초)")
                
                # API 호출
                start_time = time.time()
                result = self.recognize(chunk_path, language)
                duration = time.time() - start_time
                
                # 결과 처리 (중복 제거 개선 - 유사도 기반)
                current_text = result.get("text", "").strip()
                
                # 오버랩 중복 제거 (유사도 체크 포함)
                if i > 0 and previous_text and current_text:
                    # 이전 청크의 마지막 부분과 현재 청크의 시작 부분에서 중복 찾기
                    max_overlap = min(100, len(previous_text) // 2, len(current_text) // 2)
                    
                    # 1. 정확한 일치 찾기
                    overlap_found = False
                    for check_len in range(max_overlap, 10, -1):
                        last_part = previous_text[-check_len:]
                        
                        # 현재 텍스트에서 해당 부분 찾기
                        idx = current_text.find(last_part)
                        if idx != -1 and idx < 50:  # 시작 부분 근처에서 발견
                            # 중복 제거
                            current_text = current_text[idx + len(last_part):].strip()
                            if debug_mode:
                                print(f"🔄 청크 {i+1}: {check_len}자 정확히 일치 - 중복 제거")
                                print(f"   제거된 부분: '{last_part[:30]}...'")
                            overlap_found = True
                            break
                    
                    # 2. 유사한 패턴 찾기 (정확히 일치하지 않는 경우)
                    if not overlap_found:
                        # 단어 단위로 비교
                        prev_words = previous_text.split()
                        curr_words = current_text.split()
                        
                        if len(prev_words) >= 3 and len(curr_words) >= 3:
                            # 마지막 몇 단어와 시작 몇 단어 비교
                            for word_count in range(min(10, len(prev_words)//2), 2, -1):
                                last_words = prev_words[-word_count:]
                                
                                # 현재 텍스트의 시작 부분에서 유사한 패턴 찾기
                                for start_idx in range(min(10, len(curr_words) - word_count + 1)):
                                    curr_segment = curr_words[start_idx:start_idx + word_count]
                                    
                                    # 유사도 계산 (단어 일치율)
                                    matches = sum(1 for a, b in zip(last_words, curr_segment) if a == b)
                                    similarity = matches / word_count
                                    
                                    if similarity >= 0.7:  # 70% 이상 일치
                                        # 중복으로 판단하고 제거
                                        remove_words = start_idx + word_count
                                        remaining_words = curr_words[remove_words:]
                                        current_text = ' '.join(remaining_words).strip()
                                        
                                        if debug_mode:
                                            print(f"🔄 청크 {i+1}: {word_count}단어 유사 패턴 발견 - 중복 제거")
                                            print(f"   유사도: {similarity:.1%}")
                                            print(f"   제거된 부분: '{' '.join(curr_words[:remove_words])}'")
                                        overlap_found = True
                                        break
                                
                                if overlap_found:
                                    break
                    
                    if not overlap_found and debug_mode:
                        print(f"🔍 청크 {i+1}: 중복 없음")
                
                # 빈 텍스트가 아닌 경우만 추가
                if current_text:
                    # 결과 저장
                    chunk_result = ChunkResult(
                        chunk_index=i,
                        text=current_text,
                        success=result["success"],
                        error=result.get("error"),
                        duration=duration
                    )
                    chunks_results.append(chunk_result)
                    
                    all_texts.append(current_text)
                    previous_text = result.get("text", "").strip()  # 원본 텍스트 저장 (다음 비교용)
                else:
                    # 빈 결과 (중복 제거로 인해)
                    chunk_result = ChunkResult(
                        chunk_index=i,
                        text="[중복 제거됨]",
                        success=result["success"],
                        error=result.get("error"),
                        duration=duration
                    )
                    chunks_results.append(chunk_result)
                
                # 임시 파일 삭제
                try:
                    os.remove(chunk_path)
                except:
                    pass
                
                # 다음 요청 전 딜레이 (마지막 청크 제외)
                if i < total_chunks - 1:
                    time.sleep(self.REQUEST_DELAY)
            
            # 완료 알림
            if progress_callback:
                progress_callback(100, 100, "완료!")
            
            # 전체 텍스트 결합 (문장 단위로 개선)
            if all_texts:
                # 각 청크 텍스트 끝에 마침표가 없으면 추가
                processed_texts = []
                for i, text in enumerate(all_texts):
                    text = text.strip()
                    if text and not text[-1] in '.!?':
                        text += '.'
                    processed_texts.append(text)
                
                # 공백으로 연결 (문장 간 자연스러운 구분)
                combined_text = ' '.join(processed_texts)
                
                # 연속된 공백 제거
                combined_text = ' '.join(combined_text.split())
            else:
                combined_text = ""
            
            # 성공 여부 판단
            success_count = sum(1 for c in chunks_results if c.success)
            
            return {
                "success": success_count > 0,
                "text": combined_text,
                "chunks": chunks_results,
                "total_chunks": total_chunks,
                "success_chunks": success_count,
                "duration_seconds": duration_seconds,
                "error": None if success_count == total_chunks else f"{total_chunks - success_count}개 청크 실패"
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "chunks": [],
                "error": f"처리 중 오류: {str(e)}"
            }


# 간단한 테스트
if __name__ == "__main__":
    # 테스트용
    api_key = "YOUR_API_KEY"
    stt = ETRISpeechToText(api_key)
    result = stt.recognize("test.wav")
    
    if result["success"]:
        print(f"인식 결과: {result['text']}")
    else:
        print(f"오류: {result['error']}")