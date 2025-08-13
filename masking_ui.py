# masking_ui.py
"""
마스킹 규칙 설정 UI
"""

import streamlit as st
from typing import Dict, List
import json


def create_masking_ui(detected_entities: List[Dict]) -> Dict:
    """
    마스킹 규칙 설정 UI
    
    Args:
        detected_entities: 4단계에서 탐지된 엔티티들
    
    Returns:
        masking_rules: 사용자가 정의한 마스킹 규칙
    """
    
    st.header("🎭 마스킹 설정")
    
    # 탐지된 라벨 종류 추출
    labels = list(set(e.get("label_adjusted", e.get("label", "기타")) 
                     for e in detected_entities))
    labels.sort()
    
    # 탐지된 개별 텍스트 추출
    unique_texts = {}
    for e in detected_entities:
        text = e.get("text", "")
        label = e.get("label_adjusted", e.get("label", "기타"))
        if text and text not in unique_texts:
            unique_texts[text] = label
    
    # 마스킹 모드 선택
    mode = st.radio(
        "마스킹 모드",
        ["simple", "advanced", "custom"],
        format_func=lambda x: {
            "simple": "🟢 단순 (모든 PII 동일 처리)",
            "advanced": "🟡 고급 (라벨별 설정)",
            "custom": "🔴 커스텀 (개별 설정)"
        }[x]
    )
    
    rules = {"mode": mode}
    
    if mode == "simple":
        # 단순 모드: 하나의 규칙
        st.subheader("모든 개인정보를 다음으로 치환:")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            simple_rule = st.text_input(
                "치환 텍스트",
                value="[개인정보]",
                help="모든 PII가 이 텍스트로 치환됩니다"
            )
        with col2:
            if st.button("템플릿"):
                simple_rule = st.selectbox(
                    "템플릿 선택",
                    ["[개인정보]", "[MASKED]", "***", "XXX", "[삭제됨]"]
                )
        
        rules["simple_rule"] = simple_rule
    
    elif mode == "advanced":
        # 고급 모드: 라벨별 설정
        st.subheader("라벨별 마스킹 설정")
        
        label_rules = {}
        
        # 각 라벨에 대해 입력 필드 생성
        for label in labels:
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                st.write(f"**{label}**")
            
            with col2:
                mask_text = st.text_input(
                    f"{label} 치환",
                    key=f"label_{label}",
                    placeholder=f"[{label}]"
                )
            
            with col3:
                # 빠른 선택 버튼
                if st.button("📋", key=f"copy_{label}"):
                    mask_text = f"[{label}]"
            
            if mask_text:
                label_rules[label] = mask_text
        
        # 기본값 설정
        default = st.text_input(
            "기본 마스킹 (위에 없는 라벨용)",
            value="[기타]"
        )
        
        rules["label_rules"] = label_rules
        rules["default"] = default
    
    elif mode == "custom":
        # 커스텀 모드: 개별 텍스트 설정
        st.subheader("개별 텍스트 마스킹")
        
        # 라벨별 기본 규칙
        with st.expander("라벨별 기본 규칙"):
            label_rules = {}
            for label in labels:
                mask = st.text_input(
                    f"{label}",
                    key=f"custom_label_{label}",
                    placeholder=f"[{label}]"
                )
                if mask:
                    label_rules[label] = mask
        
        # 개별 텍스트 규칙
        st.subheader("특정 텍스트 개별 설정")
        
        entity_rules = {}
        
        # 검색/필터
        search = st.text_input("🔍 텍스트 검색", "")
        
        # 텍스트 목록 표시
        for text, label in unique_texts.items():
            if search and search.lower() not in text.lower():
                continue
            
            col1, col2, col3 = st.columns([2, 1, 3])
            
            with col1:
                st.write(f"**{text}**")
            
            with col2:
                st.caption(f"({label})")
            
            with col3:
                mask = st.text_input(
                    "치환",
                    key=f"entity_{text}",
                    placeholder="비워두면 라벨 규칙 사용"
                )
                if mask:
                    entity_rules[text] = mask
        
        # 기본값
        default = st.text_input(
            "기본 마스킹",
            value="[MASKED]",
            key="custom_default"
        )
        
        rules["label_rules"] = label_rules
        rules["entity_rules"] = entity_rules
        rules["default"] = default
    
    # 미리보기
    with st.expander("📝 설정 미리보기"):
        st.json(rules)
    
    # 프리셋 저장/불러오기
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 현재 설정 저장"):
            preset_name = st.text_input("프리셋 이름")
            if preset_name:
                # 로컬 스토리지나 파일로 저장
                with open(f"masking_preset_{preset_name}.json", "w") as f:
                    json.dump(rules, f)
                st.success(f"'{preset_name}' 저장됨")
    
    with col2:
        # 프리셋 불러오기 (구현 생략)
        pass
    
    return rules


# 통합 사용 예시
if __name__ == "__main__":
    st.set_page_config(page_title="PII 마스킹 설정", layout="wide")
    
    # 예시 엔티티 (4단계 출력)
    sample_entities = [
        {"label_adjusted": "이름", "text": "김철수", "start": 0, "end": 3},
        {"label_adjusted": "번호", "text": "010-1234-5678", "start": 10, "end": 23},
        {"label_adjusted": "주소", "text": "서울시 강남구", "start": 30, "end": 37}
    ]
    
    # UI에서 규칙 설정
    masking_rules = create_masking_ui(sample_entities)
    
    # 마스킹 실행
    if st.button("🚀 마스킹 실행"):
        from pii_masker import PIIMasker
        
        sample_text = "김철수님의 번호는 010-1234-5678이고 주소는 서울시 강남구입니다."
        
        masker = PIIMasker()
        result = masker.mask(sample_text, sample_entities, masking_rules)
        
        st.success("✅ 마스킹 완료")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("원본", result.original_text)
        with col2:
            st.text_area("마스킹", result.masked_text)
        
        st.json(result.stats)