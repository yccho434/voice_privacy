# step_by_step_pipeline_ui.py
"""
단계별 인터랙티브 파이프라인 UI
각 단계마다 사용자가 확인하고 조정할 수 있는 UI
"""

import streamlit as st
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
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

# 페이지 설정
st.set_page_config(
    page_title="음성 프라이버시 보호 시스템 - 단계별",
    page_icon="🔐",
    layout="wide"
)

# 세션 상태 초기화
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = {}
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

# 타이틀과 진행 상황
st.title("🔐 음성 프라이버시 보호 시스템")
st.markdown("단계별로 확인하고 조정하는 인터랙티브 파이프라인")

# 진행 단계 정의
STEPS = [
    {"name": "API 설정", "icon": "🔑", "key": "api_setup"},
    {"name": "음성 입력", "icon": "📤", "key": "audio_input"},
    {"name": "음성 품질 향상", "icon": "🎵", "key": "audio_enhance"},
    {"name": "음성 인식", "icon": "🎤", "key": "speech_to_text"},
    {"name": "텍스트 보정", "icon": "✏️", "key": "text_correction"},
    {"name": "개인정보 탐지", "icon": "🔍", "key": "pii_detection"},
    {"name": "마스킹 처리", "icon": "🎭", "key": "masking"},
    {"name": "최종 결과", "icon": "✅", "key": "final_result"}
]

# 진행 상황 표시
def show_progress():
    """진행 상황 시각화"""
    cols = st.columns(len(STEPS))
    for idx, (col, step) in enumerate(zip(cols, STEPS)):
        with col:
            if idx < st.session_state.current_step:
                st.success(f"{step['icon']} ~~{step['name']}~~")
            elif idx == st.session_state.current_step:
                st.info(f"**{step['icon']} {step['name']}**")
            else:
                st.text(f"{step['icon']} {step['name']}")

show_progress()
st.divider()

# 현재 단계 컨테이너
current_step = st.session_state.current_step

# Step 0: API 설정
if current_step == 0:
    st.header("🔑 Step 1: API 키 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("필수 API")
        etri_stt_key = st.text_input(
            "ETRI STT API Key",
            type="password",
            value=st.session_state.api_keys.get("etri_stt", ""),
            help="음성인식용 (필수)"
        )
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_keys.get("openai", ""),
            help="텍스트 보정용 (필수)"
        )
    
    with col2:
        st.subheader("선택 API")
        etri_ner_key = st.text_input(
            "ETRI NER API Key",
            type="password",
            value=st.session_state.api_keys.get("etri_ner", ""),
            help="개체명인식용 (선택, 없으면 정규식만 사용)"
        )
        
        st.info("💡 ETRI NER API가 없어도 정규식 기반 탐지가 작동합니다")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ API 키 확인 및 다음 단계", type="primary", use_container_width=True):
            if not etri_stt_key or not openai_key:
                st.error("필수 API 키를 모두 입력해주세요!")
            else:
                st.session_state.api_keys = {
                    "etri_stt": etri_stt_key,
                    "openai": openai_key,
                    "etri_ner": etri_ner_key
                }
                st.session_state.current_step = 1
                st.rerun()

# Step 1: 음성 입력
elif current_step == 1:
    st.header("📤 Step 2: 음성 파일 입력")
    
    uploaded_file = st.file_uploader(
        "음성 파일을 선택하세요",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="처리할 음성 파일을 업로드하세요"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            📁 파일명: {uploaded_file.name}
            📊 크기: {uploaded_file.size / (1024*1024):.1f} MB
            🎵 형식: {uploaded_file.type}
            """)
        
        with col2:
            st.audio(uploaded_file)
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_audio_path = tmp.name
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ 이전", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
        with col3:
            if st.button("다음 ➡️", type="primary", use_container_width=True):
                st.session_state.pipeline_data["audio_path"] = temp_audio_path
                st.session_state.pipeline_data["audio_name"] = uploaded_file.name
                st.session_state.current_step = 2
                st.rerun()

# Step 2: 음성 품질 향상
elif current_step == 2:
    st.header("🎵 Step 3: 음성 품질 향상")
    
    audio_path = st.session_state.pipeline_data.get("audio_path")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본 음성")
        st.audio(audio_path)
        
        st.subheader("향상 옵션")
        apply_noise_gate = st.checkbox("스펙트럴 노이즈 게이팅", value=True)
        apply_band_enhance = st.checkbox("음성 대역 강조", value=True)
        apply_normalization = st.checkbox("다이나믹 레인지 정규화", value=True)
    
    with col2:
        st.subheader("향상된 음성")
        
        if "enhanced_audio_path" in st.session_state.pipeline_data:
            st.audio(st.session_state.pipeline_data["enhanced_audio_path"])
            
            # 개선 지표 표시
            if "enhancement_metrics" in st.session_state.pipeline_data:
                metrics = st.session_state.pipeline_data["enhancement_metrics"]
                improvement = metrics.get("improvement", {})
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("노이즈 감소", f"{improvement.get('noise_reduction', 0):.1f}%")
                    st.metric("SNR 개선", f"{improvement.get('snr_improvement', 0):.1f}%")
                with col_b:
                    st.metric("다이나믹 레인지", f"{improvement.get('dynamic_range', 0):.1f}%")
                    st.metric("전체 품질", f"{improvement.get('overall', 0):.1f}%")
        
        if st.button("🔧 음성 향상 실행", type="secondary", use_container_width=True):
            with st.spinner("음성 품질 향상 중..."):
                enhancer = AudioEnhancer(target_sr=16000)
                
                output_path = tempfile.mktemp(suffix=".wav")
                enhanced_path, metrics = enhancer.enhance(
                    audio_path,
                    output_path,
                    visualize=False
                )
                
                st.session_state.pipeline_data["enhanced_audio_path"] = enhanced_path
                st.session_state.pipeline_data["enhancement_metrics"] = metrics
                st.rerun()
    
    st.divider()
    
    # 스킵 옵션
    skip_enhancement = st.checkbox("음성 향상 건너뛰기 (원본 사용)")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.button("다음 ➡️", type="primary", use_container_width=True):
            if skip_enhancement:
                st.session_state.pipeline_data["audio_for_stt"] = audio_path
            else:
                if "enhanced_audio_path" not in st.session_state.pipeline_data:
                    st.error("먼저 음성 향상을 실행해주세요!")
                else:
                    st.session_state.pipeline_data["audio_for_stt"] = st.session_state.pipeline_data["enhanced_audio_path"]
            
            if "audio_for_stt" in st.session_state.pipeline_data:
                st.session_state.current_step = 3
                st.rerun()

# Step 3: 음성 인식
# Step 3: 음성 인식
elif current_step == 3:
    st.header("🎤 Step 4: 음성 인식 (STT)")
    
    audio_for_stt = st.session_state.pipeline_data.get("audio_for_stt")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("STT 설정")
        language = st.selectbox("언어", ["korean", "english", "japanese", "chinese"])
        
        # 병렬 처리 옵션 추가
        st.divider()
        st.subheader("⚡ 처리 모드")
        
        processing_mode = st.radio(
            "처리 속도 선택",
            [
                "🔄 순차 처리 (안정적)",
                "⚡ 병렬 처리 (2배 빠름)"
            ],
            index=0,
            help="병렬 처리는 더 빠르지만 API 제한에 주의하세요"
        )
        
        # 워커 수 결정
        if "순차" in processing_mode:
            max_workers = 1
            st.info("✅ 순차 모드: 안정적이지만 느림")
        else:
            max_workers = 2
            st.success("⚡ 병렬 모드: 2워커로 약 2배 빠름")
            st.caption("테스트 결과: 100% 성공률")
        
        show_chunks = st.checkbox("청크별 결과 표시", value=True)
        
        if st.button("🎯 음성 인식 시작", type="secondary", use_container_width=True):
            with st.spinner("음성 인식 중..."):
                stt = ETRISpeechToText(st.session_state.api_keys["etri_stt"])
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                start_time = time.time()
                
                def update_progress(current, total, status):
                    progress_bar.progress(current / total)
                    status_text.text(status)
                    elapsed = time.time() - start_time
                    time_text.text(f"⏱️ 경과: {int(elapsed)}초")
                
                # 병렬 처리 옵션과 함께 실행
                result = stt.recognize_long_audio(
                    audio_for_stt,
                    language=language,
                    progress_callback=update_progress,
                    max_workers=max_workers  # 1 또는 2
                )
                
                total_time = time.time() - start_time
                
                if result["success"]:
                    st.session_state.pipeline_data["stt_result"] = result
                    st.session_state.pipeline_data["original_text"] = result["text"]
                    
                    # 처리 모드별 성공 메시지
                    mode = result.get("processing_mode", "unknown")
                    if "parallel" in mode:
                        st.success(f"✅ 병렬 처리 완료! {len(result['text'])}자 ({total_time:.1f}초)")
                    else:
                        st.success(f"✅ 순차 처리 완료! {len(result['text'])}자 ({total_time:.1f}초)")
                    
                    # 성능 비교 (예상)
                    if max_workers == 2:
                        estimated_sequential = total_time * 2
                        st.info(f"⚡ 예상 시간 절약: 약 {estimated_sequential - total_time:.1f}초")
                else:
                    st.error(f"❌ 인식 실패: {result.get('error')}")
    
    with col2:
        st.subheader("인식 결과")
        
        if "original_text" in st.session_state.pipeline_data:
            # 편집 가능한 텍스트 영역
            edited_text = st.text_area(
                "인식된 텍스트 (수정 가능)",
                value=st.session_state.pipeline_data["original_text"],
                height=300,
                help="잘못 인식된 부분을 직접 수정할 수 있습니다"
            )
            
            # 수정 사항 저장
            st.session_state.pipeline_data["edited_stt_text"] = edited_text
            
            # 통계
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("글자 수", f"{len(edited_text):,}")
            with col_b:
                st.metric("단어 수", f"{len(edited_text.split()):,}")
            with col_c:
                if "stt_result" in st.session_state.pipeline_data:
                    chunks = st.session_state.pipeline_data["stt_result"].get("total_chunks", 1)
                    st.metric("청크 수", chunks)
            with col_d:
                if "stt_result" in st.session_state.pipeline_data:
                    mode = st.session_state.pipeline_data["stt_result"].get("processing_mode", "")
                    if "parallel_2" in mode:
                        st.metric("처리 모드", "병렬 2")
                    else:
                        st.metric("처리 모드", "순차")
            
            # 청크별 결과
            if show_chunks and "stt_result" in st.session_state.pipeline_data:
                with st.expander("청크별 상세 결과"):
                    for chunk in st.session_state.pipeline_data["stt_result"].get("chunks", []):
                        if chunk.success:
                            st.text(f"청크 {chunk.chunk_index + 1}: {len(chunk.text)}자 ✅")

# Step 4: 텍스트 보정
elif current_step == 4:
    st.header("✏️ Step 5: 텍스트 보정")
    
    text_to_correct = st.session_state.pipeline_data.get("edited_stt_text", 
                                                         st.session_state.pipeline_data.get("original_text"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본 텍스트")
        st.text_area("STT 결과", value=text_to_correct, height=250, disabled=True)
        
        st.subheader("보정 옵션")
        use_context = st.checkbox("문맥 분석 사용", value=True)
        auto_punctuation = st.checkbox("문장부호 자동 추가", value=True)
        
        if st.button("✏️ 텍스트 보정 실행", type="secondary", use_container_width=True):
            with st.spinner("텍스트 보정 중..."):
                corrector = ConservativeCorrector(st.session_state.api_keys["openai"])
                correction_result = corrector.correct(text_to_correct, use_context_analysis=use_context)
                
                st.session_state.pipeline_data["correction_result"] = correction_result
                st.session_state.pipeline_data["corrected_text"] = correction_result.corrected_text
                st.rerun()
    
    with col2:
        st.subheader("보정된 텍스트")
        
        if "correction_result" in st.session_state.pipeline_data:
            correction_result = st.session_state.pipeline_data["correction_result"]
            
            # 의심 구간 표시
            if correction_result.suspicious_parts:
                st.warning(f"⚠️ {len(correction_result.suspicious_parts)}개 의심 구간 발견")
                
                with st.expander("의심 구간 검토"):
                    for i, susp in enumerate(correction_result.suspicious_parts):
                        st.write(f"**{i+1}. {susp.text}**")
                        st.write(f"이유: {susp.reason}")
                        if susp.suggestions:
                            selected = st.selectbox(
                                "선택",
                                [susp.text] + susp.suggestions,
                                key=f"susp_{i}"
                            )
                            # 선택 저장
                            if f"corrections_{i}" not in st.session_state:
                                st.session_state[f"corrections_{i}"] = selected
            
            # 편집 가능한 보정 텍스트
            final_corrected = st.text_area(
                "최종 보정 텍스트 (수정 가능)",
                value=correction_result.corrected_text,
                height=250
            )
            
            st.session_state.pipeline_data["final_corrected_text"] = final_corrected
            
            # 자동 수정 사항
            st.info(f"자동 수정: {len(correction_result.auto_corrections)}개")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col3:
        if st.button("다음 ➡️", type="primary", use_container_width=True):
            if "final_corrected_text" not in st.session_state.pipeline_data:
                st.error("먼저 텍스트 보정을 실행해주세요!")
            else:
                st.session_state.current_step = 5
                st.rerun()

# Step 5: 개인정보 탐지
elif current_step == 5:
    st.header("🔍 Step 6: 개인정보 탐지")
    
    text_to_detect = st.session_state.pipeline_data.get("final_corrected_text")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("탐지 설정")
        
        use_etri_ner = st.checkbox(
            "ETRI NER 사용",
            value=bool(st.session_state.api_keys.get("etri_ner"))
        )
        use_regex = st.checkbox("정규식 패턴 사용", value=True)
        use_keywords = st.checkbox("키워드 기반 탐지", value=True)
        
        if st.button("🔍 PII 탐지 실행", type="secondary", use_container_width=True):
            with st.spinner("개인정보 탐지 중..."):
                entities = []
                
                if use_etri_ner and st.session_state.api_keys.get("etri_ner"):
                    detector = EnhancedPIIDetector(st.session_state.api_keys["etri_ner"])
                    etri_entities = detector.detect(text_to_detect, use_spoken=True)
                    
                    # PIIEntity를 딕셔너리로 변환
                    for entity in etri_entities:
                        entities.append({
                            "text": entity.text,
                            "label": entity.label,
                            "subtype": entity.subtype,
                            "start": entity.start,
                            "end": entity.end,
                            "score": entity.score,
                            "source": entity.source,
                            "label_adjusted": entity.label
                        })
                else:
                    # 정규식 + 키워드만 사용
                    if use_regex:
                        regex_hits = detect_by_regex(text_to_detect)
                        if use_keywords:
                            entities = detect_by_keywords(text_to_detect, regex_hits)
                        else:
                            entities = regex_hits
                
                st.session_state.pipeline_data["detected_entities"] = entities
                st.success(f"✅ {len(entities)}개 개인정보 탐지")
                st.rerun()
    
    with col2:
        st.subheader("탐지 결과")
        
        if "detected_entities" in st.session_state.pipeline_data:
            entities = st.session_state.pipeline_data["detected_entities"]
            
            if entities:
                # 하이라이트 표시
                text = text_to_detect
                colors = {
                    "이름": "#FF6B6B", "번호": "#4ECDC4", "계정": "#45B7D1",
                    "주소": "#FFA07A", "소속": "#98D8C8", "금융": "#6C5CE7",
                    "URL": "#A29BFE", "신원": "#FD79A8"
                }
                
                html_parts = []
                last_end = 0
                
                for entity in sorted(entities, key=lambda x: x["start"]):
                    html_parts.append(text[last_end:entity["start"]])
                    
                    label = entity.get("label_adjusted", entity.get("label", "기타"))
                    color = colors.get(label, "#CCCCCC")
                    
                    html_parts.append(
                        f'<span style="background-color: {color}; color: white; '
                        f'padding: 2px 5px; border-radius: 3px; margin: 0 2px;">'
                        f'{entity["text"]}</span>'
                    )
                    
                    last_end = entity["end"]
                
                html_parts.append(text[last_end:])
                st.markdown("".join(html_parts), unsafe_allow_html=True)
                
                # 엔티티 편집 테이블
                st.subheader("탐지 항목 검토")
                
                # 체크박스로 선택/제외
                selected_entities = []
                for i, entity in enumerate(entities):
                    col_check, col_text, col_label = st.columns([1, 3, 2])
                    
                    with col_check:
                        include = st.checkbox("선택", value=True, key=f"entity_{i}", label_visibility="hidden")
                    
                    with col_text:
                        st.text(entity["text"])
                    
                    with col_label:
                        labels = ["이름", "번호", "주소", "계정", "금융", "URL", "신원", "소속"]
                        current_label = entity.get("label_adjusted", entity.get("label"))
                        
                        try:
                            current_index = labels.index(current_label)
                        except ValueError:
                            current_index = 0
                        
                        new_label = st.selectbox(
                            "",
                            labels,
                            index=current_index,
                            key=f"label_{i}"
                        )
                        entity["label_adjusted"] = new_label
                    
                    if include:
                        selected_entities.append(entity)
                
                st.session_state.pipeline_data["selected_entities"] = selected_entities
                st.info(f"선택된 항목: {len(selected_entities)}개")
            else:
                st.info("탐지된 개인정보가 없습니다")
                st.session_state.pipeline_data["selected_entities"] = []
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
    with col3:
        if st.button("다음 ➡️", type="primary", use_container_width=True):
            if "selected_entities" not in st.session_state.pipeline_data:
                st.error("먼저 PII 탐지를 실행해주세요!")
            else:
                st.session_state.current_step = 6
                st.rerun()

# Step 6: 마스킹 처리
elif current_step == 6:
    st.header("🎭 Step 7: 마스킹 처리")
    
    text_to_mask = st.session_state.pipeline_data.get("final_corrected_text")
    entities_to_mask = st.session_state.pipeline_data.get("selected_entities", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("마스킹 설정")
        
        masking_mode = st.selectbox(
            "마스킹 모드",
            ["simple", "advanced", "custom"],
            format_func=lambda x: {
                "simple": "단순 - 모든 PII 동일 처리",
                "advanced": "고급 - 라벨별 설정",
                "custom": "커스텀 - 개별 설정"
            }[x]
        )
        
        masking_rules = {"mode": masking_mode}
        
        if masking_mode == "simple":
            masking_rules["simple_rule"] = st.text_input(
                "모든 PII를 다음으로 치환",
                value="[개인정보]"
            )
        
        elif masking_mode == "advanced":
            st.write("라벨별 마스킹 텍스트:")
            
            label_rules = {}
            label_rules["이름"] = st.text_input("이름", value="[이름]")
            label_rules["번호"] = st.text_input("번호", value="[전화번호]")
            label_rules["주소"] = st.text_input("주소", value="[주소]")
            label_rules["계정"] = st.text_input("계정", value="[이메일]")
            label_rules["금융"] = st.text_input("금융", value="[계좌정보]")
            label_rules["URL"] = st.text_input("URL", value="[링크]")
            label_rules["신원"] = st.text_input("신원", value="[개인정보]")
            label_rules["소속"] = st.text_input("소속", value="[소속]")
            
            masking_rules["label_rules"] = label_rules
            masking_rules["default"] = st.text_input("기타", value="[기타정보]")
        
        if st.button("🎭 마스킹 적용", type="secondary", use_container_width=True):
            with st.spinner("마스킹 처리 중..."):
                masker = PIIMasker(save_mapping=True)
                
                masking_result = masker.mask(
                    text_to_mask,
                    entities_to_mask,
                    masking_rules
                )
                
                st.session_state.pipeline_data["masking_result"] = masking_result
                st.session_state.pipeline_data["masked_text"] = masking_result.masked_text
                st.success(f"✅ {masking_result.stats['total']}개 항목 마스킹 완료")
                st.rerun()
    
    with col2:
        st.subheader("마스킹 결과")
        
        if "masked_text" in st.session_state.pipeline_data:
            # 마스킹된 텍스트 표시
            masked_text = st.text_area(
                "마스킹된 텍스트 (수정 가능)",
                value=st.session_state.pipeline_data["masked_text"],
                height=300
            )
            
            st.session_state.pipeline_data["final_masked_text"] = masked_text
            
            # 마스킹 통계
            if "masking_result" in st.session_state.pipeline_data:
                result = st.session_state.pipeline_data["masking_result"]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("마스킹된 항목", result.stats["total"])
                with col_b:
                    st.metric("매핑 파일", "저장됨" if result.mapping_file else "없음")
                
                # 라벨별 통계
                if result.stats.get("by_label"):
                    st.write("라벨별 마스킹:")
                    for label, count in result.stats["by_label"].items():
                        st.write(f"- {label}: {count}개")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()
    with col3:
        if st.button("완료 ➡️", type="primary", use_container_width=True):
            if "final_masked_text" not in st.session_state.pipeline_data:
                st.error("먼저 마스킹을 적용해주세요!")
            else:
                st.session_state.current_step = 7
                st.rerun()

# Step 7: 최종 결과
elif current_step == 7:
    st.header("✅ Step 8: 최종 결과")
    
    st.success("🎉 모든 처리가 완료되었습니다!")
    
    # 전체 프로세스 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        original_text = st.session_state.pipeline_data.get("original_text", "")
        st.metric("원본 텍스트", f"{len(original_text)}자")
    
    with col2:
        entities = st.session_state.pipeline_data.get("selected_entities", [])
        st.metric("탐지된 PII", f"{len(entities)}개")
    
    with col3:
        if "masking_result" in st.session_state.pipeline_data:
            masked_count = st.session_state.pipeline_data["masking_result"].stats["total"]
            st.metric("마스킹 항목", f"{masked_count}개")
    
    with col4:
        if "enhancement_metrics" in st.session_state.pipeline_data:
            improvement = st.session_state.pipeline_data["enhancement_metrics"]["improvement"]["overall"]
            st.metric("음질 개선", f"{improvement:.1f}%")
    
    st.divider()
    
    # 단계별 결과 비교
    tabs = st.tabs(["📝 텍스트 비교", "🎵 음성 비교", "📊 통계", "💾 다운로드"])
    
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("원본 (STT)")
            st.text_area(
                "",
                value=st.session_state.pipeline_data.get("original_text", ""),
                height=400,
                disabled=True,
                key="final_original"
            )
        
        with col2:
            st.subheader("보정됨")
            st.text_area(
                "",
                value=st.session_state.pipeline_data.get("final_corrected_text", ""),
                height=400,
                disabled=True,
                key="final_corrected"
            )
        
        with col3:
            st.subheader("마스킹됨")
            st.text_area(
                "",
                value=st.session_state.pipeline_data.get("final_masked_text", ""),
                height=400,
                disabled=True,
                key="final_masked"
            )
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("원본 음성")
            if "audio_path" in st.session_state.pipeline_data:
                st.audio(st.session_state.pipeline_data["audio_path"])
        
        with col2:
            st.subheader("향상된 음성")
            if "enhanced_audio_path" in st.session_state.pipeline_data:
                st.audio(st.session_state.pipeline_data["enhanced_audio_path"])
    
    with tabs[2]:
        st.subheader("처리 통계")
        
        # 라벨별 PII 분포
        if "selected_entities" in st.session_state.pipeline_data:
            entities = st.session_state.pipeline_data["selected_entities"]
            
            label_counts = {}
            for entity in entities:
                label = entity.get("label_adjusted", entity.get("label"))
                label_counts[label] = label_counts.get(label, 0) + 1
            
            if label_counts:
                import pandas as pd
                df = pd.DataFrame(list(label_counts.items()), columns=["라벨", "개수"])
                st.bar_chart(df.set_index("라벨"))
    
    with tabs[3]:
        st.subheader("결과 다운로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "final_masked_text" in st.session_state.pipeline_data:
                st.download_button(
                    "📝 마스킹 텍스트",
                    st.session_state.pipeline_data["final_masked_text"],
                    "masked_text.txt",
                    "text/plain"
                )
        
        with col2:
            if "enhanced_audio_path" in st.session_state.pipeline_data:
                audio_path = st.session_state.pipeline_data["enhanced_audio_path"]
                if os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            "🎵 향상된 음성",
                            f.read(),
                            "enhanced_audio.wav",
                            "audio/wav"
                        )
        
        with col3:
            # 전체 결과 JSON
            result_json = json.dumps(
                st.session_state.pipeline_data,
                ensure_ascii=False,
                indent=2,
                default=str
            )
            
            st.download_button(
                "📊 전체 결과 JSON",
                result_json,
                "pipeline_result.json",
                "application/json"
            )
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ 이전 단계", use_container_width=True):
            st.session_state.current_step = 6
            st.rerun()
    
    with col3:
        if st.button("🔄 새로운 파일 처리", type="primary", use_container_width=True):
            # 세션 초기화
            st.session_state.current_step = 1
            st.session_state.pipeline_data = {}
            st.rerun()

# 사이드바에 빠른 이동
with st.sidebar:
    st.divider()
    st.subheader("🚀 빠른 이동")
    
    for idx, step in enumerate(STEPS):
        if idx <= st.session_state.current_step:
            if st.button(f"{step['icon']} {step['name']}", key=f"quick_{idx}", use_container_width=True):
                st.session_state.current_step = idx
                st.rerun()

# 푸터
st.divider()
st.caption("🔐 음성 프라이버시 보호 시스템 v2.0 - 단계별 인터랙티브 파이프라인")
st.caption("각 단계에서 결과를 확인하고 조정할 수 있습니다")