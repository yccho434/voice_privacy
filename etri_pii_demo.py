# etri_pii_demo.py
"""
ETRI NER API 중심의 PII 탐지 데모
ETRI가 메인, 정규식은 보조
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict
import pandas as pd
from collections import defaultdict

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent))

# 페이지 설정
st.set_page_config(
    page_title="ETRI NER 기반 PII 탐지",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 ETRI NER 기반 고정밀 PII 탐지")
st.markdown("ETRI 개체명 인식 API를 중심으로 한 한국어 개인정보 탐지 시스템")

# 사이드바
with st.sidebar:
    st.header("⚙️ API 설정")
    
    etri_api_key = st.text_input(
        "ETRI API Key",
        type="password",
        help="https://aiopen.etri.re.kr/ 에서 발급"
    )
    
    if etri_api_key:
        st.success("✅ API 키 입력됨")
    else:
        st.warning("⚠️ API 키를 입력해주세요")
    
    st.divider()
    
    st.subheader("🔧 탐지 설정")
    use_spoken = st.checkbox(
        "구어체 모드",
        value=False,
        help="상담 녹음, 회의록 등 구어체 텍스트용"
    )
    
    use_supplementary = st.checkbox(
        "보조 정규식 사용",
        value=True,
        help="ETRI가 놓칠 수 있는 패턴 보완"
    )
    
    st.divider()
    
    st.subheader("📊 표시 설정")
    show_raw_api = st.checkbox("API 원시 응답 보기", value=False)
    show_statistics = st.checkbox("상세 통계 보기", value=True)
    
    st.divider()
    
    # ETRI 태그 설명
    with st.expander("📚 ETRI NER 태그 목록"):
        st.markdown("""
        **주요 PII 관련 태그:**
        - `PS_NAME`: 사람 이름
        - `QT_PHONE`: 전화번호
        - `QT_ZIPCODE`: 우편번호
        - `TMI_EMAIL`: 이메일
        - `TMI_SITE`: URL
        - `OGG_ECONOMY`: 기업명
        - `OGG_EDUCATION`: 교육기관
        - `LCP_CITY`: 도시명
        - `TMM_DISEASE`: 질병명
        - `CV_OCCUPATION`: 직업
        - `DT_DAY`: 날짜
        - `QT_PRICE`: 금액
        """)

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["🔍 탐지", "📊 분석", "🧪 테스트"])

with tab1:
    # 예시 텍스트
    examples = {
        "상담 기록": """안녕하세요 김민수 고객님, KT 상담사 박지영입니다.
고객님 휴대폰 번호 010-1234-5678 확인했습니다.
서울시 강남구 테헤란로 427 위워크타워에 계신 거 맞으시죠?
이메일 minsu.kim@gmail.com으로 청구서 보내드리겠습니다.
고객님 주민등록번호 뒷자리가 1234567 맞으신가요?""",
        
        "의료 기록": """환자: 이영희 (880315-2******)
연락처: 010-9876-5432
주소: 경기도 성남시 분당구 판교역로 235
직장: 네이버 본사
진단: 고혈압, 당뇨병 2형
다음 예약: 2025년 8월 20일 오후 2시 30분
담당의: 정형외과 김철수 과장""",
        
        "금융 거래": """송금인: 최준호
계좌: 국민은행 244-25-0123456
수취인: 한미영
신한은행 110-123-456789
금액: 1,500,000원
메모: 8월 월세
거래일: 2025-08-13 15:30"""
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("예시 선택", ["직접 입력"] + list(examples.keys()))
    
    if selected != "직접 입력":
        text = st.text_area("분석할 텍스트", value=examples[selected], height=200)
    else:
        text = st.text_area("분석할 텍스트", height=200, placeholder="텍스트를 입력하세요...")
    
    if st.button("🚀 탐지 시작", type="primary", disabled=not (text and etri_api_key)):
        
        with st.spinner("ETRI API 호출 중..."):
            try:
                # ETRI 탐지기 임포트 및 실행
                from etri_ner_detector import EnhancedPIIDetector
                
                detector = EnhancedPIIDetector(etri_api_key)
                
                # 탐지 실행
                start_time = time.time()
                entities = detector.detect(text, use_spoken=use_spoken)
                elapsed = time.time() - start_time
                
                # 결과 저장
                st.session_state["entities"] = entities
                st.session_state["text"] = text
                st.session_state["elapsed_time"] = elapsed
                
                # 포맷팅
                formatted = detector.format_results(entities)
                st.session_state["formatted_results"] = formatted
                
                st.success(f"✅ 완료! {len(entities)}개 PII 탐지 (소요시간: {elapsed:.2f}초)")
                
                # API 원시 응답 (디버그용)
                if show_raw_api:
                    with st.expander("🔍 ETRI API 원시 응답"):
                        raw_result = detector.etri.analyze(text, use_spoken)
                        st.json(raw_result)
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.exception(e)
    
    # 결과 표시
    if "entities" in st.session_state:
        st.divider()
        
        entities = st.session_state["entities"]
        text = st.session_state["text"]
        
        if entities:
            # 하이라이트 표시
            st.subheader("📝 하이라이트 결과")
            
            # 색상 매핑
            colors = {
                "이름": "#FF6B6B",
                "번호": "#4ECDC4", 
                "계정": "#45B7D1",
                "주소": "#FFA07A",
                "소속": "#98D8C8",
                "금융": "#6C5CE7",
                "URL": "#A29BFE",
                "신원": "#FD79A8"
            }
            
            # HTML 생성
            html_parts = []
            last_end = 0
            
            for entity in sorted(entities, key=lambda x: x.start):
                # 이전 텍스트
                html_parts.append(text[last_end:entity.start])
                
                # 엔티티
                color = colors.get(entity.label, "#CCCCCC")
                source_icon = "🤖" if entity.source == "etri" else "🔍"
                
                html_parts.append(
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 2px 5px; border-radius: 3px; margin: 0 2px;" '
                    f'title="{entity.label} - {entity.subtype} [{entity.source}]">'
                    f'{source_icon} {entity.text}</span>'
                )
                
                last_end = entity.end
            
            html_parts.append(text[last_end:])
            
            st.markdown("".join(html_parts), unsafe_allow_html=True)
            
            # 범례
            st.caption("🤖 = ETRI NER, 🔍 = 정규식 보완")
            
            # 엔티티 테이블
            st.subheader("📋 탐지 목록")
            
            df_data = []
            for entity in entities:
                df_data.append({
                    "텍스트": entity.text,
                    "카테고리": entity.label,
                    "세부유형": entity.subtype,
                    "위치": f"{entity.start}-{entity.end}",
                    "출처": entity.source,
                    "신뢰도": f"{entity.score:.0%}"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("탐지된 개인정보가 없습니다.")

with tab2:
    st.header("📊 상세 분석")
    
    if "formatted_results" in st.session_state:
        formatted = st.session_state["formatted_results"]
        stats = formatted.get("_statistics", {})
        
        # 기본 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 PII", stats.get("total", 0))
        with col2:
            etri_count = stats.get("by_source", {}).get("etri", 0)
            st.metric("ETRI 탐지", etri_count)
        with col3:
            regex_count = stats.get("by_source", {}).get("regex", 0)
            st.metric("정규식 보완", regex_count)
        
        # 소스별 비율
        if stats.get("total", 0) > 0:
            st.subheader("탐지 소스 분석")
            col1, col2 = st.columns(2)
            
            with col1:
                # 파이 차트 데이터
                source_data = pd.DataFrame(
                    list(stats.get("by_source", {}).items()),
                    columns=["소스", "개수"]
                )
                if not source_data.empty:
                    st.bar_chart(source_data.set_index("소스"))
            
            with col2:
                # 비율 표시
                total = stats.get("total", 0)
                for source, count in stats.get("by_source", {}).items():
                    percentage = (count / total * 100) if total > 0 else 0
                    source_name = "ETRI NER" if source == "etri" else "정규식"
                    st.write(f"**{source_name}**: {count}개 ({percentage:.1f}%)")
        
        # 카테고리별 분포
        st.subheader("카테고리별 분포")
        label_data = pd.DataFrame(
            list(stats.get("by_label", {}).items()),
            columns=["카테고리", "개수"]
        )
        if not label_data.empty:
            st.bar_chart(label_data.set_index("카테고리"))
        
        # 상세 목록
        st.subheader("카테고리별 상세")
        for category, items in formatted.items():
            if category != "_statistics":
                with st.expander(f"{category} ({len(items)}개)"):
                    for item in items:
                        source_emoji = "🤖" if item["source"] == "etri" else "🔍"
                        st.write(f"{source_emoji} **{item['text']}** - 위치: {item['position']}, 신뢰도: {item['confidence']}")
        
        # 처리 시간
        if "elapsed_time" in st.session_state:
            st.divider()
            st.metric("처리 시간", f"{st.session_state['elapsed_time']:.3f}초")
    else:
        st.info("먼저 텍스트를 분석해주세요.")

with tab3:
    st.header("🧪 API 테스트")
    
    test_text = st.text_input("테스트 문장", value="김철수의 전화번호는 010-1234-5678입니다.")
    
    if st.button("API 직접 호출", disabled=not etri_api_key):
        with st.spinner("API 호출 중..."):
            try:
                from etri_ner_detector import ETRILanguageAnalyzer
                
                analyzer = ETRILanguageAnalyzer(etri_api_key)
                result = analyzer.analyze(test_text, use_spoken)
                
                st.subheader("API 응답")
                st.json(result)
                
                # NER 결과 파싱
                if result:
                    st.subheader("파싱된 개체명")
                    for sentence in result.get("sentence", []):
                        st.write(f"**문장**: {sentence.get('text', '')}")
                        
                        for ne in sentence.get("NE", []):
                            ne_type = ne.get("type", "")
                            ne_text = ne.get("text", "")
                            
                            # PII 관련 여부 체크
                            is_pii = ne_type in analyzer.TAG_TO_PII
                            
                            if is_pii:
                                label, subtype = analyzer.TAG_TO_PII[ne_type]
                                st.success(f"✅ PII 탐지: '{ne_text}' → {label} ({subtype}) [태그: {ne_type}]")
                            else:
                                st.info(f"ℹ️ 기타 개체: '{ne_text}' [태그: {ne_type}]")
                
            except Exception as e:
                st.error(f"오류: {str(e)}")

# 푸터
st.divider()
st.caption("🔒 ETRI NER API 기반 한국어 PII 탐지 시스템")
st.caption("일일 5,000건 제한 | 1회 최대 1만 글자")