# audio_to_text_ui.py
"""
음성→텍스트 변환 Streamlit UI
UX 최적화 버전
"""

import streamlit as st
import os
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent))

from audio_to_text_pipeline import AudioToTextPipeline, PipelineConfig

# 페이지 설정
st.set_page_config(
    page_title="음성→텍스트 변환기",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# 커스텀 CSS
st.markdown("""
<style>
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.error-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# 헤더
st.title("🎙️ 음성→텍스트 변환기")
st.markdown("고품질 음성 전처리 + ETRI STT 통합 솔루션")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키
    api_key = st.text_input(
        "ETRI API 키",
        type="password",
        help="https://aiopen.etri.re.kr 에서 발급",
        key="api_key"
    )
    
    if api_key:
        st.success("✅ API 키 입력됨")
    else:
        st.warning("⚠️ API 키를 입력하세요")
    
    st.divider()
    
    # 처리 옵션
    st.subheader("🎵 음성 전처리")
    
    enhance_audio = st.checkbox(
        "음성 품질 향상",
        value=True,
        help="노이즈 제거 및 음성 대역 최적화"
    )
    
    if enhance_audio:
        col1, col2 = st.columns(2)
        with col1:
            aggressive = st.checkbox(
                "강력 노이즈 제거",
                value=False,
                help="노이즈가 심한 경우"
            )
        with col2:
            auto_detect = st.checkbox(
                "자동 노이즈 탐지",
                value=True,
                help="무음 구간 자동 분석"
            )
    else:
        aggressive = False
        auto_detect = False
    
    st.divider()
    
    # STT 옵션
    st.subheader("🎤 음성 인식")
    
    enable_timestamps = st.checkbox(
        "타임스탬프 포함",
        value=True,
        help="각 문장 시작 시간 표시 [MM:SS]"
    )
    
    parallel_mode = st.checkbox(
        "병렬 처리",
        value=True,
        help="2배 빠른 처리 (긴 오디오)"
    )
    
    if parallel_mode:
        max_workers = 2
        st.info("⚡ 병렬 모드: 2워커 사용")
    else:
        max_workers = 1
        st.info("🔄 순차 모드: 안정적 처리")
    
    st.divider()
    
    # 저장 옵션
    st.subheader("💾 저장 옵션")
    
    save_enhanced = st.checkbox(
        "향상된 음성 저장",
        value=False
    )
    
    save_transcript = st.checkbox(
        "텍스트 파일 저장",
        value=True
    )
    
    output_dir = st.text_input(
        "저장 경로",
        value="./output",
        help="결과 파일 저장 위치"
    )

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["📤 파일 업로드", "📊 처리 결과", "📈 통계"])

with tab1:
    # 파일 업로드
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "음성 파일 선택",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            accept_multiple_files=True,
            help="여러 파일 동시 처리 가능"
        )
        
        if uploaded_files:
            # 파일 정보 표시
            file_info = []
            total_size = 0
            
            for file in uploaded_files:
                size_mb = file.size / (1024 * 1024)
                total_size += size_mb
                file_info.append({
                    "파일명": file.name,
                    "크기": f"{size_mb:.1f} MB",
                    "형식": file.type.split('/')[-1].upper()
                })
            
            df = pd.DataFrame(file_info)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 요약
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("총 파일 수", f"{len(uploaded_files)}개")
            with col_b:
                st.metric("총 크기", f"{total_size:.1f} MB")
            with col_c:
                # 예상 시간 계산 (대략적)
                est_time_sec = total_size * 5  # MB당 5초 추정
                if parallel_mode:
                    est_time_sec /= 2
                est_time = str(timedelta(seconds=int(est_time_sec)))
                st.metric("예상 시간", est_time)
    
    with col2:
        # 빠른 테스트
        st.subheader("🧪 빠른 테스트")
        
        test_file = "test.mp3"
        if os.path.exists(test_file):
            if st.button("테스트 파일 사용", use_container_width=True):
                st.info(f"🧪 {test_file} 로드됨")
                # 테스트 파일을 세션에 저장
                st.session_state.use_test_file = True
        else:
            st.caption("test.mp3 파일 없음")
    
    # 처리 시작 버튼
    st.divider()
    
    if uploaded_files or st.session_state.get('use_test_file'):
        if st.button(
            "🚀 변환 시작",
            type="primary",
            use_container_width=True,
            disabled=not api_key or st.session_state.processing
        ):
            if not api_key:
                st.error("❌ API 키를 입력하세요!")
            else:
                st.session_state.processing = True
                
                # 파이프라인 설정
                config = PipelineConfig(
                    etri_api_key=api_key,
                    enhance_audio=enhance_audio,
                    aggressive_denoise=aggressive,
                    auto_detect_noise=auto_detect,
                    parallel_stt=parallel_mode,
                    max_workers=max_workers,
                    enable_timestamps=enable_timestamps,  # 타임스탬프 옵션 추가
                    save_enhanced_audio=save_enhanced,
                    save_transcript=save_transcript,
                    output_dir=output_dir
                )
                
                pipeline = AudioToTextPipeline(config)
                
                # 처리할 파일 준비
                files_to_process = []
                
                if st.session_state.get('use_test_file'):
                    files_to_process.append(test_file)
                    st.session_state.use_test_file = False
                else:
                    # 업로드된 파일을 임시 저장
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=Path(uploaded_file.name).suffix
                        ) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            files_to_process.append(tmp.name)
                
                # 진행 상황 컨테이너
                progress_container = st.container()
                
                with progress_container:
                    # 진행 바
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    eta_text = st.empty()
                    
                    # 상세 정보
                    with st.expander("📊 처리 상세", expanded=True):
                        detail_text = st.empty()
                        metrics_container = st.container()
                    
                    # 진행 콜백
                    def update_progress(step, percent, message, eta):
                        progress_bar.progress(percent / 100)
                        status_text.text(f"💫 {message}")
                        
                        if eta:
                            eta_text.text(f"⏱️ 예상 시간: {eta}")
                        
                        # 상세 정보 업데이트
                        detail_text.text(f"""
단계: {step}
진행률: {percent}%
상태: {message}
                        """)
                    
                    # 처리 시작
                    results = []
                    start_time = time.time()
                    
                    for i, audio_path in enumerate(files_to_process):
                        file_name = Path(audio_path).name
                        
                        # 파일별 진행
                        def file_progress(step, percent, message, eta):
                            overall_percent = int((i / len(files_to_process)) * 100 + 
                                                percent / len(files_to_process))
                            msg = f"[{i+1}/{len(files_to_process)}] {file_name}: {message}"
                            update_progress(step, overall_percent, msg, eta)
                        
                        # 처리 실행
                        result = pipeline.process(audio_path, file_progress)
                        results.append(result)
                        
                        # 결과 저장
                        # 오디오 데이터를 bytes로 변환하여 저장
                        audio_data = None
                        if not st.session_state.get('use_test_file'):
                            # uploaded_file.getbuffer()는 memoryview를 반환하므로 bytes()로 변환
                            audio_data = bytes(uploaded_file.getbuffer())
                        
                        st.session_state.results_history.append({
                            'timestamp': result.timestamp,
                            'file_name': file_name,
                            'success': result.success,
                            'transcript_length': len(result.transcript),
                            'processing_time': result.processing_time,
                            'audio_duration': result.audio_duration,
                            'improvement': result.audio_improvement,
                            'has_timestamps': result.formatted_transcript is not None,
                            'sentence_count': len(result.sentences) if result.sentences else 0,
                            'result': result,
                            'original_audio_data': audio_data,
                            'audio_file_name': uploaded_file.name if not st.session_state.get('use_test_file') else 'test.mp3'
                        })
                        
                        # 중간 결과 표시
                        with metrics_container:
                            if result.success:
                                st.success(f"✅ {file_name}: 완료")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("처리 시간", f"{result.processing_time:.1f}초")
                                with col2:
                                    st.metric("텍스트", f"{len(result.transcript)}자")
                                with col3:
                                    if result.audio_improvement:
                                        noise_reduction = result.audio_improvement.get('noise_reduction', 0)
                                        st.metric("노이즈 감소", f"{noise_reduction:.1f}%")
                            else:
                                st.error(f"❌ {file_name}: 실패")
                    
                    # 완료
                    total_time = time.time() - start_time
                    progress_bar.progress(100)
                    status_text.text("✅ 모든 처리 완료!")
                    eta_text.text(f"총 소요 시간: {timedelta(seconds=int(total_time))}")
                    
                    # 임시 파일 정리
                    if not st.session_state.get('use_test_file'):
                        for tmp_path in files_to_process:
                            try:
                                os.remove(tmp_path)
                            except:
                                pass
                
                st.session_state.processing = False
                st.balloons()
    
    else:
        st.info("👆 음성 파일을 업로드하거나 테스트 파일을 사용하세요")

with tab2:
    st.header("📊 처리 결과")
    
    if st.session_state.results_history:
        # 최근 결과
        latest_results = st.session_state.results_history[-5:][::-1]
        
        for result_info in latest_results:
            result = result_info['result']
            
            with st.expander(
                f"📁 {result_info['file_name']} - {result_info['timestamp']}",
                expanded=True
            ):
                if result.success:
                    # 🎵 음성 재생 섹션 추가
                    st.subheader("🎧 음성 재생")
                    
                    # 음성 데이터 확인
                    audio_played = False
                    
                    # 1. 세션에 저장된 원본 오디오 데이터 확인
                    if result_info.get('original_audio_data'):
                        col_audio1, col_audio2 = st.columns([3, 1])
                        
                        with col_audio1:
                            st.caption(f"🔊 원본 음성: {result_info.get('audio_file_name', 'audio.wav')}")
                            st.audio(result_info['original_audio_data'], format='audio/wav')
                            audio_played = True
                        
                        with col_audio2:
                            st.info("💡 팁")
                            st.caption("• 음성을 들으면서 아래 텍스트를 수정하세요")
                            st.caption("• 재생 바를 드래그하여 특정 부분으로 이동 가능")
                    
                    # 2. 향상된 음성 파일 확인
                    elif result.enhanced_audio_path and os.path.exists(result.enhanced_audio_path):
                        col_audio1, col_audio2 = st.columns([3, 1])
                        
                        with col_audio1:
                            st.caption("🔊 향상된 음성")
                            with open(result.enhanced_audio_path, 'rb') as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format='audio/wav')
                            audio_played = True
                        
                        with col_audio2:
                            st.info("💡 팁")
                            st.caption("• 음성을 들으면서 아래 텍스트를 수정하세요")
                            st.caption("• 재생 바를 드래그하여 특정 부분으로 이동 가능")
                    
                    # 3. test.mp3 파일 확인
                    elif result_info.get('file_name') == 'test.mp3' and os.path.exists('test.mp3'):
                        col_audio1, col_audio2 = st.columns([3, 1])
                        
                        with col_audio1:
                            st.caption("🔊 테스트 음성: test.mp3")
                            with open('test.mp3', 'rb') as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format='audio/mp3')
                            audio_played = True
                        
                        with col_audio2:
                            st.info("💡 팁")
                            st.caption("• 음성을 들으면서 아래 텍스트를 수정하세요")
                            st.caption("• 재생 바를 드래그하여 특정 부분으로 이동 가능")
                    
                    # 4. 원본 경로에서 시도
                    elif result.original_audio_path and os.path.exists(result.original_audio_path):
                        col_audio1, col_audio2 = st.columns([3, 1])
                        
                        with col_audio1:
                            st.caption("🔊 원본 음성")
                            with open(result.original_audio_path, 'rb') as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format='audio/wav')
                            audio_played = True
                        
                        with col_audio2:
                            st.info("💡 팁")
                            st.caption("• 음성을 들으면서 아래 텍스트를 수정하세요")
                            st.caption("• 재생 바를 드래그하여 특정 부분으로 이동 가능")
                    
                    if not audio_played:
                        st.warning("⚠️ 음성 파일을 찾을 수 없습니다. 새로 업로드한 파일만 재생 가능합니다.")
                    
                    st.divider()
                    
                    # 메트릭
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("처리 시간", f"{result.processing_time:.1f}초")
                    with col2:
                        st.metric("오디오 길이", f"{result.audio_duration:.1f}초")
                    with col3:
                        st.metric("텍스트", f"{len(result.transcript)}자")
                    with col4:
                        speed = result.audio_duration / result.processing_time if result.processing_time > 0 else 0
                        st.metric("처리 속도", f"{speed:.1f}x")
                    
                    # 음성 개선 정보
                    if result.audio_improvement:
                        with st.expander("🎵 음성 개선 상세"):
                            imp = result.audio_improvement
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"노이즈 감소: {imp.get('noise_reduction', 0):.1f}%")
                            with col2:
                                st.info(f"명료도: {imp.get('clarity', 0):.1f}%")
                            with col3:
                                st.info(f"전체 개선: {imp.get('overall', 0):.1f}%")
                    
                    # STT 통계
                    if result.stt_stats:
                        with st.expander("📈 인식 통계"):
                            stats = result.stt_stats
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total = stats.get('total_chunks', 1)
                                success = stats.get('success_chunks', 1)
                                st.metric("청크 성공률", f"{success}/{total}")
                            with col2:
                                retry = stats.get('retry_count', 0)
                                st.metric("재시도", f"{retry}회")
                            with col3:
                                rate = stats.get('success_rate', 1.0)
                                st.metric("성공률", f"{rate*100:.0f}%")
                    
                    st.divider()
                    
                    # 변환된 텍스트 (편집 가능)
                    st.subheader("✏️ 텍스트 편집")
                    
                    # 편집 모드 선택
                    col_edit1, col_edit2 = st.columns([2, 1])
                    
                    with col_edit1:
                        # 타임스탬프 포함/미포함 선택
                        if result.formatted_transcript:
                            text_mode = st.radio(
                                "편집 형식",
                                ["타임스탬프 포함", "일반 텍스트"],
                                horizontal=True,
                                key=f"text_mode_{result.timestamp}"
                            )
                            
                            if text_mode == "타임스탬프 포함":
                                display_text = result.formatted_transcript
                            else:
                                display_text = result.transcript
                        else:
                            display_text = result.transcript
                    
                    with col_edit2:
                        # 텍스트 통계
                        st.caption(f"글자 수: {len(display_text)}")
                        st.caption(f"줄 수: {len(display_text.splitlines())}")
                    
                    # 편집 가능한 텍스트 영역 (더 크게)
                    edited_text = st.text_area(
                        "텍스트를 수정할 수 있습니다:",
                        value=display_text,
                        height=300,
                        key=f"text_edit_{result.timestamp}",
                        help="음성을 들으면서 잘못 인식된 부분을 수정하세요"
                    )
                    
                    # 수정 사항 확인
                    if edited_text != display_text:
                        st.success("✏️ 텍스트가 수정되었습니다")
                        
                        # 변경 사항 미리보기
                        with st.expander("변경 사항 보기"):
                            col_orig, col_edited = st.columns(2)
                            with col_orig:
                                st.caption("원본")
                                st.text(display_text[:200] + "..." if len(display_text) > 200 else display_text)
                            with col_edited:
                                st.caption("수정본")
                                st.text(edited_text[:200] + "..." if len(edited_text) > 200 else edited_text)
                    
                    st.divider()
                    
                    # 다운로드 버튼 (수정된 텍스트 포함)
                    st.subheader("💾 다운로드")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # 수정된 텍스트 다운로드
                        st.download_button(
                            "📝 수정된 텍스트",
                            edited_text,
                            f"transcript_edited_{result.timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            type="primary"
                        )
                    
                    with col2:
                        # 원본 텍스트 다운로드
                        st.download_button(
                            "📄 원본 텍스트",
                            display_text,
                            f"transcript_original_{result.timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col3:
                        # 음성 다운로드 (저장된 데이터 우선 사용)
                        if result_info.get('original_audio_data'):
                            # 세션에 저장된 원본 데이터 사용
                            st.download_button(
                                "🎵 원본 음성",
                                result_info['original_audio_data'],
                                f"audio_{result.timestamp}_{result_info.get('audio_file_name', 'audio.wav')}",
                                mime="audio/wav",
                                use_container_width=True
                            )
                        elif result.enhanced_audio_path and os.path.exists(result.enhanced_audio_path):
                            with open(result.enhanced_audio_path, 'rb') as f:
                                st.download_button(
                                    "🎵 향상된 음성",
                                    f.read(),
                                    f"enhanced_{result.timestamp}.wav",
                                    mime="audio/wav",
                                    use_container_width=True
                                )
                        elif result_info.get('file_name') == 'test.mp3' and os.path.exists('test.mp3'):
                            with open('test.mp3', 'rb') as f:
                                st.download_button(
                                    "🎵 테스트 음성",
                                    f.read(),
                                    f"test_{result.timestamp}.mp3",
                                    mime="audio/mp3",
                                    use_container_width=True
                                )
                        elif result.original_audio_path and os.path.exists(result.original_audio_path):
                            with open(result.original_audio_path, 'rb') as f:
                                st.download_button(
                                    "🎵 원본 음성",
                                    f.read(),
                                    f"audio_{result.timestamp}{Path(result.original_audio_path).suffix}",
                                    mime="audio/wav",
                                    use_container_width=True
                                )
                    
                    with col4:
                        # 메타데이터 JSON (수정 이력 포함)
                        meta = {
                            'timestamp': result.timestamp,
                            'file_name': result_info['file_name'],
                            'success': result.success,
                            'original_transcript': result.transcript,
                            'edited_transcript': edited_text if edited_text != display_text else None,
                            'formatted_transcript': result.formatted_transcript,
                            'sentences': result.sentences,
                            'processing_time': result.processing_time,
                            'audio_duration': result.audio_duration,
                            'improvement': result.audio_improvement,
                            'stt_stats': result.stt_stats,
                            'was_edited': edited_text != display_text
                        }
                        
                        st.download_button(
                            "📊 메타데이터",
                            json.dumps(meta, ensure_ascii=False, indent=2),
                            f"meta_{result.timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # SRT 자막 다운로드 (타임스탬프가 있는 경우)
                    if result.sentences:
                        st.divider()
                        
                        # SRT 형식 생성
                        srt_lines = []
                        for i, sentence in enumerate(result.sentences, 1):
                            start_time = sentence['start_time']
                            end_time = sentence.get('end_time', start_time + 3)
                            
                            # SRT 시간 형식
                            def format_srt_time(seconds):
                                hours = int(seconds // 3600)
                                minutes = int((seconds % 3600) // 60)
                                secs = int(seconds % 60)
                                millis = int((seconds % 1) * 1000)
                                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                            
                            srt_lines.append(str(i))
                            srt_lines.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                            srt_lines.append(sentence['text'])
                            srt_lines.append("")
                        
                        srt_content = "\n".join(srt_lines)
                        
                        st.download_button(
                            "🎬 SRT 자막 파일",
                            srt_content,
                            f"subtitle_{result.timestamp}.srt",
                            mime="text/plain",
                            use_container_width=True,
                            help="비디오 자막용 SRT 파일"
                        )
                
                else:
                    st.error("처리 실패")
                    if result.stt_stats.get('error'):
                        st.code(result.stt_stats['error'])
        
        # 전체 기록 삭제
        st.divider()
        if st.button("🗑️ 전체 기록 삭제", type="secondary"):
            if st.button("정말로 삭제하시겠습니까?", type="secondary", key="confirm_delete"):
                st.session_state.results_history = []
                st.rerun()
    
    else:
        st.info("아직 처리된 결과가 없습니다")

with tab3:
    st.header("📈 처리 통계")
    
    if st.session_state.results_history:
        # 전체 통계 계산
        total_files = len(st.session_state.results_history)
        success_files = sum(1 for r in st.session_state.results_history if r['success'])
        total_time = sum(r['processing_time'] for r in st.session_state.results_history)
        total_audio = sum(r['audio_duration'] for r in st.session_state.results_history)
        total_text = sum(r['transcript_length'] for r in st.session_state.results_history)
        
        # 요약 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 처리 파일", f"{total_files}개")
            st.metric("성공률", f"{(success_files/total_files*100):.0f}%")
        
        with col2:
            st.metric("총 처리 시간", f"{timedelta(seconds=int(total_time))}")
            st.metric("총 오디오 길이", f"{timedelta(seconds=int(total_audio))}")
        
        with col3:
            st.metric("총 텍스트", f"{total_text:,}자")
            avg_text = total_text / total_files if total_files > 0 else 0
            st.metric("평균 텍스트", f"{avg_text:.0f}자")
        
        with col4:
            speed = total_audio / total_time if total_time > 0 else 0
            st.metric("평균 처리 속도", f"{speed:.1f}x")
            
            # 시간당 처리량
            throughput = total_audio / total_time * 3600 if total_time > 0 else 0
            st.metric("시간당 처리", f"{throughput/60:.0f}분")
        
        # 처리 이력 차트
        st.divider()
        st.subheader("📊 처리 이력")
        
        # 데이터프레임 생성
        df_history = pd.DataFrame([
            {
                '시간': r['timestamp'],
                '파일': r['file_name'],
                '성공': '✅' if r['success'] else '❌',
                '처리시간(초)': r['processing_time'],
                '오디오(초)': r['audio_duration'],
                '텍스트(자)': r['transcript_length'],
                '속도(x)': r['audio_duration'] / r['processing_time'] if r['processing_time'] > 0 else 0
            }
            for r in st.session_state.results_history
        ])
        
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        # 성능 차트
        if len(df_history) > 1:
            st.divider()
            st.subheader("📉 성능 추이")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(df_history[['처리시간(초)', '오디오(초)']])
            
            with col2:
                st.bar_chart(df_history[['텍스트(자)']])
    
    else:
        st.info("통계를 표시할 데이터가 없습니다")

# 푸터
st.divider()
st.caption("🎙️ 음성→텍스트 변환기 v2.0 | librosa + ETRI STT")
st.caption("노이즈 제거 • 음성 향상 • 병렬 처리 • 스마트 청킹")