# integrated_pipeline_ui.py
"""
통합 파이프라인 Streamlit UI
"""

import streamlit as st
import os
import time
from pathlib import Path
from integrated_pipeline import IntegratedPipeline, PipelineConfig, PipelineResult
from pii_masker import create_masking_ui

# 페이지 설정
st.set_page_config(
    page_title="음성 프라이버시 보호 시스템",
    page_icon="🔐",
    layout="wide"
)

# 타이틀
st.title("🔐 음성 프라이버시 보호 통합 시스템")
st.markdown("음성 파일의 개인정보를 자동으로 탐지하고 마스킹하는 End-to-End 파이프라인")

# 세션 상태 초기화
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# 사이드바 - API 키 설정
with st.sidebar:
    st.header("🔑 API 설정")
    
    etri_stt_key = st.text_input(
        "ETRI STT API Key",
        type="password",
        help="음성인식용"
    )
    
    etri_ner_key = st.text_input(
        "ETRI NER API Key", 
        type="password",
        help="개체명인식용 (선택)"
    )
    
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="텍스트 보정용"
    )
    
    st.divider()
    
    st.header("⚙️ 파이프라인 설정")
    
    enhance_audio = st.checkbox("음성 품질 향상", value=True)
    use_context = st.checkbox("문맥 분석 사용", value=True)
    use_etri_ner = st.checkbox("ETRI NER 사용", value=bool(etri_ner_key))
    save_intermediate = st.checkbox("중간 결과 저장", value=True)
    
    st.divider()
    
    # 진행 상태
    if st.session_state.current_step > 1:
        st.header("📊 진행 상태")
        progress = (st.session_state.current_step - 1) / 5
        st.progress(progress)
        
        steps = ["음성 향상", "음성 인식", "텍스트 보정", "PII 탐지", "마스킹"]
        for i, step in enumerate(steps, 1):
            if i < st.session_state.current_step:
                st.success(f"✅ {step}")
            elif i == st.session_state.current_step:
                st.info(f"🔄 {step}")
            else:
                st.text(f"⏳ {step}")

# 메인 영역 - 단계별 UI
tabs = st.tabs(["📤 입력", "🔧 처리", "📊 결과"])

with tabs[0]:
    st.header("1️⃣ 음성 파일 입력")
    
    uploaded_file = st.file_uploader(
        "음성 파일 선택",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="처리할 음성 파일을 업로드하세요"
    )
    
    if uploaded_file:
        # 파일 정보
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
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_audio_path = tmp.name
            st.session_state.audio_path = temp_audio_path

with tabs[1]:
    st.header("2️⃣ 파이프라인 실행")
    
    if "audio_path" not in st.session_state:
        st.warning("먼저 음성 파일을 업로드하세요")
    else:
        # 마스킹 규칙 설정
        st.subheader("마스킹 규칙 설정")
        
        # 간단한 설정
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
            
            col1, col2 = st.columns(2)
            
            label_rules = {}
            with col1:
                label_rules["이름"] = st.text_input("이름", value="[이름]")
                label_rules["번호"] = st.text_input("전화번호", value="[전화번호]")
                label_rules["주소"] = st.text_input("주소", value="[주소]")
                label_rules["계정"] = st.text_input("이메일", value="[이메일]")
            
            with col2:
                label_rules["금융"] = st.text_input("계좌정보", value="[계좌정보]")
                label_rules["URL"] = st.text_input("URL", value="[링크]")
                label_rules["신원"] = st.text_input("신원정보", value="[개인정보]")
                label_rules["소속"] = st.text_input("소속", value="[소속]")
            
            masking_rules["label_rules"] = label_rules
            masking_rules["default"] = st.text_input("기타", value="[기타정보]")
        
        # 실행 버튼
        if st.button("🚀 파이프라인 실행", type="primary"):
            
            # API 키 확인
            if not etri_stt_key or not openai_key:
                st.error("필수 API 키를 입력하세요 (ETRI STT, OpenAI)")
            else:
                # 설정 생성
                config = PipelineConfig(
                    etri_stt_key=etri_stt_key,
                    etri_ner_key=etri_ner_key or "",
                    openai_key=openai_key,
                    enhance_audio=enhance_audio,
                    use_context_analysis=use_context,
                    use_etri_ner=use_etri_ner and bool(etri_ner_key),
                    save_intermediate=save_intermediate
                )
                
                # 파이프라인 생성
                pipeline = IntegratedPipeline(config)
                
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(step, total, message):
                    progress_bar.progress(step / total)
                    status_text.text(message)
                    st.session_state.current_step = step
                
                # 실행
                with st.spinner("처리 중..."):
                    result = pipeline.process(
                        st.session_state.audio_path,
                        masking_rules,
                        update_progress
                    )
                
                st.session_state.pipeline_result = result
                
                if result.errors:
                    st.error(f"오류 발생: {result.errors}")
                else:
                    st.success("✅ 파이프라인 완료!")
                    st.balloons()

with tabs[2]:
    st.header("3️⃣ 처리 결과")
    
    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        
        # 요약 정보
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("원본 텍스트", f"{len(result.original_text)}자")
        with col2:
            st.metric("탐지된 PII", f"{result.pii_detection.get('total_entities', 0)}개")
        with col3:
            total_time = sum(result.processing_time.values())
            st.metric("처리 시간", f"{total_time:.1f}초")
        
        st.divider()
        
        # 단계별 결과
        st.subheader("📝 단계별 결과")
        
        # 1. 음성 향상
        with st.expander("1️⃣ 음성 품질 향상"):
            if result.audio_enhancement:
                improvement = result.audio_enhancement.get("improvement", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("노이즈 감소", f"{improvement.get('noise_reduction', 0):.1f}%")
                    st.metric("SNR 개선", f"{improvement.get('snr_improvement', 0):.1f}%")
                
                with col2:
                    st.metric("다이나믹 레인지", f"{improvement.get('dynamic_range', 0):.1f}%")
                    st.metric("전체 품질", f"{improvement.get('overall', 0):.1f}%")
                
                if result.enhanced_audio_path:
                    st.audio(result.enhanced_audio_path)
        
        # 2. STT 결과
        with st.expander("2️⃣ 음성 인식"):
            st.text_area("인식된 텍스트", result.original_text, height=150)
            
            if result.speech_to_text:
                chunks = result.speech_to_text.get("total_chunks", 1)
                st.info(f"청크 수: {chunks}")
        
        # 3. 텍스트 보정
        with st.expander("3️⃣ 텍스트 보정"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area("원본", result.original_text, height=100)
            
            with col2:
                st.text_area("보정", result.corrected_text, height=100)
            
            if result.text_correction:
                suspicious = result.text_correction.get("suspicious_parts", 0)
                if suspicious > 0:
                    st.warning(f"의심 구간: {suspicious}개")
        
        # 4. PII 탐지
        with st.expander("4️⃣ PII 탐지"):
            if result.pii_detection:
                entities = result.pii_detection.get("entities", [])
                
                # 라벨별 통계
                label_counts = {}
                for e in entities:
                    label = e.get("label_adjusted", e.get("label", "기타"))
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                st.bar_chart(label_counts)
                
                # 엔티티 목록
                for label, count in label_counts.items():
                    st.write(f"**{label}**: {count}개")
        
        # 5. 최종 마스킹 결과
        with st.expander("5️⃣ 마스킹 결과", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area("보정된 텍스트", result.corrected_text, height=200)
            
            with col2:
                st.text_area("마스킹된 텍스트", result.masked_text, height=200)
            
            if result.masking:
                st.info(f"마스킹된 항목: {result.masking.get('total_masked', 0)}개")
                
                if result.masking.get("mapping_file"):
                    st.success(f"매핑 파일: {result.masking['mapping_file']}")
        
        # 다운로드
        st.divider()
        st.subheader("💾 다운로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📝 마스킹 텍스트",
                result.masked_text,
                f"masked_{result.session_id}.txt",
                "text/plain"
            )
        
        with col2:
            if result.enhanced_audio_path and os.path.exists(result.enhanced_audio_path):
                with open(result.enhanced_audio_path, "rb") as f:
                    st.download_button(
                        "🎵 향상된 음성",
                        f.read(),
                        f"enhanced_{result.session_id}.wav",
                        "audio/wav"
                    )
        
        with col3:
            # 전체 결과 JSON
            import json
            from dataclasses import asdict
            
            result_json = json.dumps(
                asdict(result),
                ensure_ascii=False,
                indent=2,
                default=str
            )
            
            st.download_button(
                "📊 전체 결과",
                result_json,
                f"result_{result.session_id}.json",
                "application/json"
            )
    else:
        st.info("파이프라인을 실행하면 결과가 여기에 표시됩니다.")

# 푸터
st.divider()
st.caption("🔐 음성 프라이버시 보호 시스템 v1.0 - 심리상담 녹음 데이터 특화")