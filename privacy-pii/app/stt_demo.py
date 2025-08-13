# stt_demo_fixed.py
"""
ETRI 음성인식 + PII 탐지 Streamlit 데모
test.mp3 파일 자동 로드 지원
"""

import streamlit as st
import tempfile
import os
import sys
import time
import json
import io
from pathlib import Path
from datetime import datetime

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / "privacy-pii" / "app"))

from speech_to_text import ETRISpeechToText
from detectors.regex_patterns import detect_by_regex
from detectors.keyword_rules import detect_by_keywords

# pydub 체크
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="음성인식 + PII 탐지 데모",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 음성인식 + PII 탐지 데모")
st.markdown("ETRI 음성인식 API를 사용하여 음성을 텍스트로 변환하고 개인정보를 탐지합니다.")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    api_key = st.text_input(
        "ETRI API Key",
        type="password",
        help="ETRI에서 발급받은 API 키를 입력하세요"
    )
    
    language = st.selectbox(
        "언어 선택",
        options=["korean", "english", "japanese", "chinese"],
        index=0
    )
    
    st.divider()
    st.subheader("🔧 디버그 옵션")
    debug_mode = st.checkbox("디버그 모드", value=False)
    show_chunks = st.checkbox("청크별 결과 표시", value=True)
    show_raw_response = st.checkbox("원시 API 응답 표시", value=False)
    save_debug_log = st.checkbox("디버그 로그 저장", value=False)
    
    st.divider()
    st.subheader("PII 탐지 옵션")
    enable_pii = st.checkbox("PII 탐지 활성화", value=True)
    
    request_delay = st.number_input("요청 간 딜레이(초)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 음성 입력")
    
    # 테스트 파일 옵션 (보조 기능)
    test_file_path = "test.mp3"
    test_file_exists = os.path.exists(test_file_path)
    
    if test_file_exists:
        use_test = st.checkbox(f"🧪 빠른 테스트 ({test_file_path} 사용)", value=False)
    else:
        use_test = False
    
    # 메인 기능: 파일 업로드
    if not use_test:
        uploaded_file = st.file_uploader(
            "음성 파일을 업로드하세요",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="20초 이하는 즉시 처리, 긴 파일은 자동 분할 처리"
        )
        
        if uploaded_file:
            file_info = {
                "name": uploaded_file.name,
                "size_mb": uploaded_file.size / (1024 * 1024),
                "type": uploaded_file.type
            }
            
            st.info(f"""
            📁 파일: {file_info['name']}
            📊 크기: {file_info['size_mb']:.1f} MB
            🎵 형식: {file_info['type']}
            ⚡ 처리: {'자동 분할' if PYDUB_AVAILABLE else '20초 이하만'}
            """)
            
            st.audio(uploaded_file, format=file_info['type'])
    
    # 테스트 모드
    else:
        st.info("🧪 테스트 모드: test.mp3 파일 사용")
        
        with open(test_file_path, "rb") as f:
            test_bytes = f.read()
        
        uploaded_file = io.BytesIO(test_bytes)
        uploaded_file.name = test_file_path
        
        file_info = {
            "name": test_file_path,
            "size_mb": len(test_bytes) / (1024 * 1024),
            "type": "audio/mp3"
        }
        
        st.info(f"""
        📁 파일: {file_info['name']}
        📊 크기: {file_info['size_mb']:.1f} MB
        🎵 형식: MP3
        """)
        
        st.audio(test_bytes, format="audio/mp3")
    
    # STT 실행 버튼
    if uploaded_file and st.button("🎯 음성 인식 시작", type="primary", disabled=not api_key):
        if not api_key:
            st.error("❌ API 키를 입력해주세요!")
        else:
            # 임시 파일로 저장
            suffix = Path(file_info['name']).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                # BytesIO인지 UploadedFile인지 확인
                if hasattr(uploaded_file, 'read'):
                    # BytesIO (테스트 파일)
                    uploaded_file.seek(0)
                    tmp_file.write(uploaded_file.read())
                else:
                    # UploadedFile (업로드 파일)
                    tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # STT 실행
                stt = ETRISpeechToText(api_key)
                stt.REQUEST_DELAY = request_delay
                
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                start_time = time.time()
                
                def update_progress(current, total, status):
                    progress_bar.progress(current / total)
                    status_text.text(f"📊 {status}")
                    elapsed = time.time() - start_time
                    time_text.text(f"⏱️ 경과: {int(elapsed)}초")
                
                # 음성 인식 실행 (debug_mode 전달)
                if debug_mode:
                    st.info("🔧 디버그 모드 활성화")
                    debug_container = st.container()
                
                result = stt.recognize_long_audio(tmp_path, language, update_progress, debug_mode=debug_mode)
                
                if result["success"]:
                    st.session_state["stt_result"] = result["text"]
                    st.session_state["stt_metadata"] = result
                    
                    # 결과 요약
                    total_time = time.time() - start_time
                    if result.get('total_chunks', 1) > 1:
                        st.success(f"""
                        ✅ 완료!
                        - 청크: {result['success_chunks']}/{result['total_chunks']} 성공
                        - 처리 시간: {int(total_time)}초
                        - 텍스트: {len(result['text'])}자
                        """)
                    else:
                        st.success(f"✅ 완료! ({int(total_time)}초)")
                    
                    # 청크별 결과 (상세)
                    if show_chunks and result.get('chunks'):
                        with st.expander("📦 청크별 결과", expanded=debug_mode):
                            for chunk in result['chunks']:
                                col_idx, col_status, col_text = st.columns([1, 2, 4])
                                
                                with col_idx:
                                    st.write(f"청크 {chunk.chunk_index + 1}")
                                
                                with col_status:
                                    if chunk.success:
                                        if chunk.text == "[중복 제거됨]":
                                            st.warning("중복")
                                        else:
                                            st.success(f"{len(chunk.text)}자")
                                    else:
                                        st.error("실패")
                                
                                with col_text:
                                    if chunk.success and chunk.text != "[중복 제거됨]":
                                        # 청크 텍스트 미리보기
                                        preview = chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
                                        st.text(preview)
                                    elif not chunk.success:
                                        st.error(chunk.error)
                    
                    # 원시 응답 (디버그)
                    if show_raw_response and debug_mode:
                        with st.expander("🔍 원시 API 응답"):
                            debug_data = {
                                "success": result["success"],
                                "total_chunks": result.get("total_chunks"),
                                "success_chunks": result.get("success_chunks"),
                                "duration_seconds": result.get("duration_seconds"),
                                "text_length": len(result.get("text", "")),
                                "processing_time": total_time,
                                "chunks_detail": [
                                    {
                                        "index": c.chunk_index,
                                        "success": c.success,
                                        "text_length": len(c.text) if c.success else 0,
                                        "error": c.error,
                                        "duration": c.duration
                                    } for c in result.get("chunks", [])
                                ]
                            }
                            st.json(debug_data)
                    
                    # 디버그 로그 저장
                    if save_debug_log and debug_mode:
                        log_data = {
                            "timestamp": datetime.now().isoformat(),
                            "file": file_info['name'],
                            "result": result,
                            "processing_time": total_time
                        }
                        log_file = f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        
                        st.download_button(
                            "💾 디버그 로그 다운로드",
                            data=json.dumps(log_data, ensure_ascii=False, indent=2, default=str),
                            file_name=log_file,
                            mime="application/json"
                        )
                else:
                    st.error(f"❌ 실패: {result.get('error')}")
                    if debug_mode and result.get("chunks"):
                        with st.expander("오류 상세"):
                            for chunk in result["chunks"]:
                                if not chunk.success:
                                    st.error(f"청크 {chunk.chunk_index + 1}: {chunk.error}")
            
            finally:
                os.unlink(tmp_path)

with col2:
    st.header("📝 인식 결과")
    
    if "stt_result" in st.session_state and st.session_state["stt_result"]:
        text = st.session_state["stt_result"]
        
        # 텍스트 표시
        edited_text = st.text_area("인식된 텍스트", value=text, height=150)
        
        # 통계
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("글자 수", f"{len(edited_text):,}")
        with col_b:
            st.metric("단어 수", f"{len(edited_text.split()):,}")
        with col_c:
            meta = st.session_state.get("stt_metadata", {})
            st.metric("청크 수", f"{meta.get('total_chunks', 1)}")
        
        # PII 탐지
        if enable_pii:
            st.divider()
            st.subheader("🔍 PII 탐지 결과")
            
            regex_hits = detect_by_regex(edited_text)
            keyword_hits = detect_by_keywords(edited_text, regex_hits)
            
            if keyword_hits:
                st.metric("탐지된 개인정보", f"{len(keyword_hits)}개")
                
                # 하이라이트
                colors = {
                    "금융": "#4CAF50",
                    "계정": "#607D8B",
                    "URL": "#8E24AA",
                    "번호": "#3F51B5",
                    "주소": "#FFC107",
                    "이름": "#EF5350",
                    "소속": "#009688",
                    "신원": "#90A4AE",
                }
                
                html = []
                last_end = 0
                
                for span in sorted(keyword_hits, key=lambda x: x["start"]):
                    html.append(edited_text[last_end:span["start"]])
                    
                    label = span.get("label_adjusted", span.get("label", ""))
                    color = colors.get(label, "#CCCCCC")
                    entity_text = edited_text[span["start"]:span["end"]]
                    
                    html.append(
                        f'<span style="background-color: {color}; '
                        f'color: white; padding: 2px 4px; '
                        f'border-radius: 3px;">{entity_text}</span>'
                    )
                    
                    last_end = span["end"]
                
                html.append(edited_text[last_end:])
                st.markdown("".join(html), unsafe_allow_html=True)
                
                # 상세 정보
                with st.expander("탐지 상세"):
                    for i, hit in enumerate(keyword_hits, 1):
                        label = hit.get("label_adjusted", hit.get("label", ""))
                        entity = edited_text[hit["start"]:hit["end"]]
                        st.write(f"{i}. **{label}**: {entity}")
            else:
                st.info("탐지된 개인정보가 없습니다.")
    else:
        st.info("👈 음성 파일을 업로드하고 인식을 시작하세요.")

# 하단 정보
with st.expander("ℹ️ 사용 방법"):
    st.markdown("""
    1. **API 키 입력**: 사이드바에 ETRI API 키 입력
    2. **파일 선택**: test.mp3 사용 또는 파일 업로드
    3. **음성 인식**: '음성 인식 시작' 클릭
    4. **결과 확인**: 텍스트 및 PII 탐지 결과 확인
    
    **테스트 파일**: 프로젝트 폴더에 'test.mp3' 넣으면 자동 인식
    """)