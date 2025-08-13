# conservative_review_demo.py
"""
보수적 보정 + 사용자 검토 Streamlit UI
GPT는 확실한 것만, 애매한 건 사용자가
"""

import streamlit as st
import os
from conservative_corrector import (
    ConservativeCorrector, 
    InteractiveReviewer,
    conservative_correct_with_review
)

# 페이지 설정
st.set_page_config(
    page_title="STT 텍스트 검토 시스템",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ STT 텍스트 보정 및 검토 시스템")
st.markdown("**GPT는 확실한 것만 수정, 애매한 것은 사용자가 결정**")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="GPT API 키"
    )
    
    st.divider()
    
    st.subheader("🎯 보정 옵션")
    
    auto_punctuation = st.checkbox("문장부호 자동 추가", value=True)
    auto_spacing = st.checkbox("띄어쓰기 자동 수정", value=True)
    highlight_suspicious = st.checkbox("의심 구간 하이라이트", value=True)
    
    st.divider()
    
    with st.expander("📚 시스템 설명"):
        st.markdown("""
        **자동 수정 (GPT)**
        - 문장부호 (? . , !)
        - 중복 조사 (을를 → 를)
        - 명백한 띄어쓰기
        
        **사용자 검토 필요**
        - 발음 혼동 의심 (더 친 → 다친?)
        - 문맥상 이상한 부분
        - 애매한 표현
        
        **Temperature: 0.1** (매우 보수적)
        """)

# 메인 영역
tab1, tab2, tab3 = st.tabs(["📝 보정", "👀 검토", "📊 결과"])

with tab1:
    st.header("1️⃣ STT 텍스트 입력")
    
    # 예시 텍스트
    examples = {
        "의료 상담": """최근에 건강은 어때 건강한 것 같아요. 최근에 더 친 곳 있어? 
아빠가 꽉 잡아서 어깨가 아파요. 언제 다쳤어 이번 주 월요일이야""",
        
        "일반 대화": """어제 회의에서 논의한 내용 정리해서 보내드릴게요
네 감사합니다 내일까지 검토하고 피드백 드리겠습니다""",
        
        "전화 상담": """네 안녕하세요 김민수님 오늘 어떻게 지내셨어요
음 사실 요즘 너무 우울하고 불안해서요 잠도 잘 못자고"""
    }
    
    selected = st.selectbox("예시 선택", ["직접 입력"] + list(examples.keys()))
    
    if selected != "직접 입력":
        input_text = st.text_area(
            "원본 STT 텍스트",
            value=examples[selected],
            height=150
        )
    else:
        input_text = st.text_area(
            "원본 STT 텍스트",
            height=150,
            placeholder="STT 출력 텍스트를 입력하세요..."
        )
    
    if st.button("🚀 보정 시작", type="primary", disabled=not (input_text and api_key)):
        with st.spinner("보수적 보정 중..."):
            result = conservative_correct_with_review(input_text, api_key)
            st.session_state['correction_result'] = result
            
            # 결과 요약
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("자동 수정", result['auto_corrections'])
            
            with col2:
                review_count = len(result.get('review_items', []))
                st.metric("검토 필요", f"{review_count}개")
            
            with col3:
                status = "✅ 완료" if not result['needs_review'] else "⚠️ 검토 필요"
                st.metric("상태", status)
            
            if result['needs_review']:
                st.warning(f"⚠️ {review_count}개 항목이 검토가 필요합니다. '검토' 탭에서 확인하세요.")
            else:
                st.success("✅ 모든 보정이 완료되었습니다.")

with tab2:
    st.header("2️⃣ 사용자 검토")
    
    if 'correction_result' in st.session_state:
        result = st.session_state['correction_result']
        
        if result['needs_review']:
            st.info(f"🔍 {len(result['review_items'])}개 항목을 검토해주세요.")
            
            # 전체 텍스트 표시 (하이라이트 포함)
            st.subheader("보정된 텍스트")
            
            # 하이라이트 처리
            display_text = result['corrected']
            if highlight_suspicious:
                for item in result['review_items']:
                    text_to_highlight = item['text']
                    display_text = display_text.replace(
                        text_to_highlight,
                        f"**🟨 [{text_to_highlight}]**"
                    )
            
            st.markdown(display_text)
            
            st.divider()
            
            # 개별 검토 항목
            st.subheader("검토 항목")
            
            user_choices = {}
            
            for i, item in enumerate(result['review_items']):
                with st.expander(f"📍 항목 {i+1}: '{item['text']}'", expanded=True):
                    st.write(f"**이유:** {item['reason']}")
                    st.write(f"**신뢰도:** {item['confidence']:.0%} (낮음)")
                    
                    # 선택 옵션
                    options = [item['text']] + item['suggestions'] + ["직접 입력"]
                    
                    choice = st.radio(
                        "선택하세요:",
                        options,
                        key=f"choice_{i}",
                        help="원본을 유지하거나 대안을 선택하세요"
                    )
                    
                    # 직접 입력
                    if choice == "직접 입력":
                        custom = st.text_input(
                            "직접 입력:",
                            key=f"custom_{i}"
                        )
                        if custom:
                            user_choices[item['text']] = custom
                    elif choice != item['text']:
                        user_choices[item['text']] = choice
            
            # 적용 버튼
            st.divider()
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("✅ 선택 적용", type="primary"):
                    reviewer = InteractiveReviewer()
                    
                    # 간단한 Result 객체 생성
                    from conservative_corrector import CorrectionResult
                    corr_result = CorrectionResult(
                        original_text=result['original'],
                        corrected_text=result['corrected'],
                        suspicious_parts=[],
                        auto_corrections=[],
                        needs_review=False
                    )
                    
                    final_text = reviewer.apply_user_corrections(corr_result, user_choices)
                    st.session_state['final_text'] = final_text
                    st.success("✅ 적용 완료! '결과' 탭에서 확인하세요.")
            
            with col2:
                if st.button("🔄 원본 유지"):
                    st.session_state['final_text'] = result['corrected']
                    st.info("원본이 유지되었습니다.")
            
        else:
            st.success("✅ 검토가 필요한 항목이 없습니다.")
            st.session_state['final_text'] = result['corrected']
    else:
        st.info("먼저 텍스트를 보정해주세요.")

with tab3:
    st.header("3️⃣ 최종 결과")
    
    if 'correction_result' in st.session_state:
        result = st.session_state['correction_result']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 원본 (STT)")
            st.text_area(
                "",
                value=result['original'],
                height=200,
                disabled=True,
                key="original_display"
            )
        
        with col2:
            st.subheader("✨ 최종 결과")
            
            final = st.session_state.get('final_text', result['corrected'])
            st.text_area(
                "",
                value=final,
                height=200,
                disabled=True,
                key="final_display"
            )
        
        # 변경 사항 요약
        st.divider()
        st.subheader("📊 처리 요약")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **자동 수정**
            {result['auto_corrections']}
            """)
        
        with col2:
            review_count = len(result.get('review_items', []))
            st.info(f"""
            **사용자 검토**
            {review_count}개 항목
            """)
        
        with col3:
            # 원본과 최종 비교
            original_len = len(result['original'])
            final_len = len(final)
            preservation = (1 - abs(final_len - original_len) / original_len) * 100
            
            st.info(f"""
            **원본 보존율**
            {preservation:.1f}%
            """)
        
        # 다운로드 버튼
        st.divider()
        
        st.download_button(
            label="💾 최종 텍스트 다운로드",
            data=final,
            file_name="corrected_text.txt",
            mime="text/plain"
        )
        
        # 다음 단계 안내
        st.divider()
        st.success("""
        ✅ **보정 완료!** 
        
        이제 다음 단계로 진행할 수 있습니다:
        - NER (개체명 인식)
        - 개인정보 탐지
        - 마스킹 처리
        """)
    else:
        st.info("보정을 먼저 실행해주세요.")

# 푸터
st.divider()
st.caption("💡 보수적 보정 시스템 - GPT는 확실한 것만, 애매한 건 사용자가")
st.caption("Temperature: 0.1 | 최소 수정 원칙")