"""
병렬 처리 테스트 스크립트
python test_parallel_stt.py 로 실행
"""

import os
import sys
import time
import json
import base64
import urllib3
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

# pydub 임포트
try:
    from pydub import AudioSegment
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    PYDUB_AVAILABLE = True
except:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub 설치 필요: pip install pydub imageio-ffmpeg")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class TestResult:
    """테스트 결과"""
    mode: str
    total_chunks: int
    success_count: int
    fail_count: int
    total_time: float
    avg_time_per_chunk: float
    errors: List[str]
    

class ParallelSTTTester:
    """병렬 처리 테스터"""
    
    API_URL = "http://epretx.etri.re.kr:8000/api/WiseASR_Recognition"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http = urllib3.PoolManager()
        self.test_results = []
        
    def create_test_chunks(self, audio_path: str, num_chunks: int = 5, 
                          chunk_seconds: int = 10) -> List[str]:
        """테스트용 청크 생성"""
        if not PYDUB_AVAILABLE:
            print("❌ pydub가 필요합니다")
            return []
            
        print(f"\n📦 테스트 청크 생성 중... ({num_chunks}개, 각 {chunk_seconds}초)")
        
        audio = AudioSegment.from_file(audio_path)
        chunk_ms = chunk_seconds * 1000
        chunks = []
        temp_dir = tempfile.mkdtemp()
        
        for i in range(num_chunks):
            start_ms = i * chunk_ms
            end_ms = min(start_ms + chunk_ms, len(audio))
            
            if start_ms >= len(audio):
                break
                
            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join(temp_dir, f"test_chunk_{i:03d}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            print(f"  ✅ 청크 {i+1}: {start_ms/1000:.1f}초 ~ {end_ms/1000:.1f}초")
        
        return chunks
    
    def recognize_single(self, chunk_path: str, chunk_index: int = 0) -> Dict:
        """단일 청크 인식"""
        try:
            with open(chunk_path, "rb") as f:
                audio_contents = base64.b64encode(f.read()).decode("utf8")
            
            request_json = {
                "argument": {
                    "language_code": "korean",
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
                body=json.dumps(request_json)
            )
            elapsed = time.time() - start_time
            
            if response.status == 200:
                result = json.loads(response.data.decode("utf-8"))
                if result.get("result", -1) == 0:
                    return {
                        "success": True,
                        "chunk_index": chunk_index,
                        "time": elapsed,
                        "text": result.get("return_object", {}).get("recognized", "")
                    }
            
            return {
                "success": False,
                "chunk_index": chunk_index,
                "time": elapsed,
                "error": f"API Error: {response.status}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "chunk_index": chunk_index,
                "time": 0,
                "error": str(e)
            }
    
    def test_sequential(self, chunks: List[str], delay: float = 1.0) -> TestResult:
        """순차 처리 테스트"""
        print(f"\n🔄 순차 처리 테스트 (딜레이: {delay}초)")
        print("-" * 50)
        
        results = []
        errors = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            print(f"  처리 중: 청크 {i+1}/{len(chunks)}...", end="")
            result = self.recognize_single(chunk, i)
            results.append(result)
            
            if result["success"]:
                print(f" ✅ 성공 ({result['time']:.2f}초)")
            else:
                print(f" ❌ 실패: {result['error']}")
                errors.append(f"청크 {i}: {result['error']}")
            
            # 마지막 청크가 아니면 대기
            if i < len(chunks) - 1:
                time.sleep(delay)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            mode=f"순차 (딜레이 {delay}초)",
            total_chunks=len(chunks),
            success_count=success_count,
            fail_count=len(chunks) - success_count,
            total_time=total_time,
            avg_time_per_chunk=total_time / len(chunks),
            errors=errors
        )
    
    def test_parallel(self, chunks: List[str], max_workers: int = 2, 
                     batch_delay: float = 0) -> TestResult:
        """병렬 처리 테스트"""
        print(f"\n⚡ 병렬 처리 테스트 (워커: {max_workers}개, 배치 딜레이: {batch_delay}초)")
        print("-" * 50)
        
        results = []
        errors = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업 제출
            future_to_chunk = {}
            
            for i, chunk in enumerate(chunks):
                # 배치 딜레이 (옵션)
                if batch_delay > 0 and i > 0 and i % max_workers == 0:
                    print(f"  💤 배치 딜레이 {batch_delay}초...")
                    time.sleep(batch_delay)
                
                future = executor.submit(self.recognize_single, chunk, i)
                future_to_chunk[future] = i
                print(f"  📤 청크 {i+1} 제출")
            
            # 결과 수집
            print("\n  ⏳ 결과 대기 중...")
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                    
                    if result["success"]:
                        print(f"  ✅ 청크 {chunk_idx+1} 완료 ({result['time']:.2f}초)")
                    else:
                        print(f"  ❌ 청크 {chunk_idx+1} 실패: {result['error']}")
                        errors.append(f"청크 {chunk_idx}: {result['error']}")
                        
                except Exception as e:
                    print(f"  ❌ 청크 {chunk_idx+1} 예외: {e}")
                    errors.append(f"청크 {chunk_idx}: {str(e)}")
                    results.append({
                        "success": False,
                        "chunk_index": chunk_idx,
                        "error": str(e)
                    })
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            mode=f"병렬 (워커 {max_workers}개)",
            total_chunks=len(chunks),
            success_count=success_count,
            fail_count=len(chunks) - success_count,
            total_time=total_time,
            avg_time_per_chunk=total_time / len(chunks),
            errors=errors
        )
    
    def test_smart_parallel(self, chunks: List[str]) -> TestResult:
        """스마트 병렬 처리 (적응형)"""
        print(f"\n🧠 스마트 병렬 처리 테스트 (적응형)")
        print("-" * 50)
        
        results = []
        errors = []
        start_time = time.time()
        max_workers = 1  # 1개로 시작
        
        # 첫 청크 테스트
        print("  🔍 첫 청크로 테스트...")
        first_result = self.recognize_single(chunks[0], 0)
        results.append(first_result)
        
        if not first_result["success"]:
            print("  ⚠️ 첫 청크 실패, 순차 모드 유지")
            # 나머지 순차 처리
            for i, chunk in enumerate(chunks[1:], 1):
                time.sleep(1.0)
                result = self.recognize_single(chunk, i)
                results.append(result)
        else:
            print(f"  ✅ 첫 청크 성공! 병렬 처리 시작")
            
            # 2개씩 시도
            max_workers = 2
            remaining = chunks[1:]
            
            for batch_start in range(0, len(remaining), max_workers):
                batch = remaining[batch_start:batch_start + max_workers]
                batch_indices = list(range(batch_start + 1, batch_start + 1 + len(batch)))
                
                print(f"\n  📦 배치 처리: 청크 {batch_indices}")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self.recognize_single, chunk, idx) 
                              for chunk, idx in zip(batch, batch_indices)]
                    
                    batch_results = []
                    for future in futures:
                        result = future.result()
                        batch_results.append(result)
                        results.append(result)
                
                # 성공률 체크
                batch_success = sum(1 for r in batch_results if r["success"])
                success_rate = batch_success / len(batch_results)
                
                print(f"  📊 배치 성공률: {success_rate:.0%}")
                
                if success_rate < 0.5:
                    print(f"  ⚠️ 성공률 낮음, 워커 감소: {max_workers} → 1")
                    max_workers = 1
                elif success_rate == 1.0 and max_workers < 3:
                    print(f"  🚀 성공률 100%, 워커 증가: {max_workers} → {max_workers + 1}")
                    max_workers += 1
                
                # 배치 간 대기
                if batch_start + max_workers < len(remaining):
                    time.sleep(1.5)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r["success"])
        
        return TestResult(
            mode="스마트 병렬 (적응형)",
            total_chunks=len(chunks),
            success_count=success_count,
            fail_count=len(chunks) - success_count,
            total_time=total_time,
            avg_time_per_chunk=total_time / len(chunks),
            errors=errors
        )
    
    def run_all_tests(self, audio_path: str, num_chunks: int = 5):
        """모든 테스트 실행"""
        print("\n" + "="*60)
        print("🧪 ETRI STT 병렬 처리 테스트")
        print("="*60)
        
        # 테스트 청크 생성
        chunks = self.create_test_chunks(audio_path, num_chunks, chunk_seconds=10)
        
        if not chunks:
            print("❌ 청크 생성 실패")
            return
        
        print(f"\n✅ {len(chunks)}개 청크 준비 완료")
        
        # 테스트 실행
        tests = [
            ("순차 (딜레이 1초)", lambda: self.test_sequential(chunks, delay=1.0)),
            ("순차 (딜레이 0.5초)", lambda: self.test_sequential(chunks, delay=0.5)),
            ("병렬 (2 워커)", lambda: self.test_parallel(chunks, max_workers=2)),
            ("병렬 (3 워커)", lambda: self.test_parallel(chunks, max_workers=3)),
            ("스마트 병렬", lambda: self.test_smart_parallel(chunks)),
        ]
        
        results = []
        
        for name, test_func in tests:
            print(f"\n{'='*60}")
            input(f"Enter를 눌러 '{name}' 테스트 시작...")
            
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                print(f"❌ 테스트 실패: {e}")
                continue
            
            # 테스트 간 충분한 대기
            print(f"\n⏸️ 다음 테스트까지 3초 대기...")
            time.sleep(3)
        
        # 결과 요약
        self.print_summary(results)
        
        # 청크 파일 정리
        for chunk in chunks:
            try:
                os.remove(chunk)
            except:
                pass
    
    def print_summary(self, results: List[TestResult]):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 테스트 결과 요약")
        print("="*60)
        
        # 테이블 헤더
        print(f"{'모드':<25} {'성공/전체':<12} {'총시간':<10} {'청크당':<10} {'성공률':<8}")
        print("-"*75)
        
        for r in results:
            success_rate = (r.success_count / r.total_chunks * 100) if r.total_chunks > 0 else 0
            print(f"{r.mode:<25} {r.success_count}/{r.total_chunks:<10} "
                  f"{r.total_time:<10.2f}초 {r.avg_time_per_chunk:<10.2f}초 "
                  f"{success_rate:<8.1f}%")
            
            if r.errors:
                print(f"  ⚠️ 오류: {len(r.errors)}개")
                for error in r.errors[:2]:  # 처음 2개만 표시
                    print(f"    - {error}")
        
        # 최적 모드 추천
        if results:
            best = min(results, key=lambda x: x.total_time if x.success_count == x.total_chunks else float('inf'))
            if best.success_count == best.total_chunks:
                print(f"\n🏆 추천 모드: {best.mode}")
                print(f"   - 모든 청크 성공, 최단 시간: {best.total_time:.2f}초")
        
        # 결과 저장
        self.save_results(results)
    
    def save_results(self, results: List[TestResult]):
        """결과를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parallel_test_result_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "results": [
                {
                    "mode": r.mode,
                    "total_chunks": r.total_chunks,
                    "success_count": r.success_count,
                    "fail_count": r.fail_count,
                    "total_time": r.total_time,
                    "avg_time_per_chunk": r.avg_time_per_chunk,
                    "errors": r.errors
                }
                for r in results
            ]
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {filename}")


def main():
    """메인 실행 함수"""
    print("🔧 ETRI STT 병렬 처리 테스터")
    print("-" * 40)
    
    # API 키 입력
    api_key = input("ETRI API 키 입력 (또는 Enter로 환경변수 사용): ").strip()
    if not api_key:
        api_key = os.getenv("ETRI_API_KEY")
        if not api_key:
            print("❌ API 키가 필요합니다!")
            return
    
    # 오디오 파일 선택
    audio_path = input("테스트할 오디오 파일 경로 (기본: test.mp3): ").strip()
    if not audio_path:
        audio_path = "test.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return
    
    # 청크 개수
    num_chunks = input("테스트할 청크 개수 (기본: 5): ").strip()
    num_chunks = int(num_chunks) if num_chunks else 5
    
    # 테스터 실행
    tester = ParallelSTTTester(api_key)
    tester.run_all_tests(audio_path, num_chunks)


if __name__ == "__main__":
    main()