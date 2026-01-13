"""
Streamlit UI - AI Image Filter Pipeline
"""

import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import time
from datetime import datetime

# ============ 설정 ============
API_URL = "http://localhost:8000/api/v1"  # FastAPI 서버 주소

# 페이지 설정
st.set_page_config(
    page_title="AI Image Filter Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CSS 스타일 ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verdict-ai {
        background-color: #ffcccb;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .verdict-real {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .verdict-uncertain {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # 헤더
    st.markdown('<p class="main-header">🔍 AI Image Filter Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML 학습 데이터셋에서 AI 생성 이미지를 필터링하는 3-Layer 검증 시스템</p>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        api_url = st.text_input("API URL", value=API_URL)
        skip_ai_detection = st.checkbox("AI 탐지 스킵 (빠른 분석)", value=False)
        
        st.divider()
        
        st.header("📊 분석 파이프라인")
        st.markdown("""
        **Layer 1**: DinoV2 Hash Check
        - facebook/dinov2-small 벡터 유사도

        **Layer 2**: Metadata Analysis
        - EXIF 진위성 점수
        - EXIF 비정상 패턴 탐지
        - C2PA Content Credentials
        - AI 도구 시그니처

        **Layer 3**: AI Detection
        - [HuggingFace Model](https://huggingface.co/dima806/ai_vs_human_generated_image_detection)
        """)
        st.info("ℹ️ Stateless 모드 - 데이터베이스를 사용하지 않습니다. 모든 분석은 실시간으로만 처리됩니다.")
        
        st.divider()
        

    
    # 메인 탭
    tab1, tab2 = st.tabs(["📤 단일 이미지 분석", "📦 배치 분석"])
    
    # ============ 탭 1: 단일 이미지 분석 ============
    with tab1:
        st.header("단일 이미지 분석")
        
        uploaded_file = st.file_uploader(
            "이미지를 업로드하세요",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            key="single_upload"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 업로드된 이미지")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                st.caption(f"파일명: {uploaded_file.name} | 크기: {uploaded_file.size:,} bytes")
            
            with col2:
                st.subheader("🔬 분석 결과")
                
                if st.button("🚀 분석 시작", type="primary", key="analyze_single"):
                    with st.spinner("분석 중..."):
                        try:
                            # API 호출
                            uploaded_file.seek(0)
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            params = {"skip_ai_detection": skip_ai_detection}
                            
                            response = requests.post(
                                f"{api_url}/analyze",
                                files=files,
                                params=params,
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                display_result(result)
                            else:
                                st.error(f"분석 실패: {response.text}")
                                
                        except requests.exceptions.ConnectionError:
                            st.error("⚠️ API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
                            st.info("로컬 테스트: `uvicorn app.main:app --reload`")
                        except Exception as e:
                            st.error(f"오류 발생: {e}")
    
    # ============ 탭 2: 배치 분석 ============
    with tab2:
        st.header("배치 이미지 분석")
        st.info("최대 50개 이미지를 한 번에 분석할 수 있습니다.")
        
        uploaded_files = st.file_uploader(
            "여러 이미지를 업로드하세요",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)}개 파일 선택됨")
            
            if st.button("🚀 배치 분석 시작", type="primary", key="analyze_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"분석 중: {file.name} ({i+1}/{len(uploaded_files)})")
                    
                    try:
                        file.seek(0)
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        params = {"skip_ai_detection": skip_ai_detection}
                        
                        response = requests.post(
                            f"{api_url}/analyze",
                            files=files,
                            params=params,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            metadata = result.get("metadata_result", {})
                            hash_res = result.get("hash_result", {})
                            results.append({
                                "파일명": file.name,
                                "판정": result.get("final_verdict", "unknown"),
                                "확신도": f"{result.get('confidence_score', 0):.1%}",
                                "DinoV2 유사도": f"{hash_res.get('similarity', 0):.1%}",
                                "EXIF 진위성": f"{metadata.get('exif_authenticity_score', 0):.2f}",
                                "AI 시그니처": ", ".join(metadata.get("ai_tool_signatures", [])) or "-",
                                "EXIF 비정상": len(metadata.get("exif_inconsistencies", []))
                            })
                        else:
                            results.append({
                                "파일명": file.name,
                                "판정": "error",
                                "확신도": "-",
                                "DinoV2 유사도": "-",
                                "EXIF 진위성": "-",
                                "AI 시그니처": "-",
                                "EXIF 비정상": "-"
                            })
                    except Exception as e:
                        results.append({
                            "파일명": file.name,
                            "판정": "error",
                            "확신도": "-",
                            "DinoV2 유사도": "-",
                            "EXIF 진위성": "-",
                            "AI 시그니처": str(e)[:30],
                            "EXIF 비정상": "-"
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ 분석 완료!")
                
                # 결과 테이블
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # 통계
                col1, col2, col3 = st.columns(3)
                ai_count = sum(1 for r in results if r["판정"] == "ai_generated")
                real_count = sum(1 for r in results if r["판정"] == "likely_real")
                uncertain_count = sum(1 for r in results if r["판정"] == "uncertain")
                
                col1.metric("🤖 AI 생성", ai_count)
                col2.metric("✅ 실제 이미지", real_count)
                col3.metric("❓ 불확실", uncertain_count)
                
                # CSV 다운로드
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 결과 CSV 다운로드",
                    csv,
                    f"ai_filter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
    



def display_result(result: dict):
    """분석 결과 표시"""
    verdict = result.get("final_verdict", "unknown")
    confidence = result.get("confidence_score", 0)
    
    # 판정 결과 표시
    if verdict == "ai_generated":
        st.markdown(f"""
        <div class="verdict-ai">
            <h3>🤖 AI 생성 이미지로 판정</h3>
            <p>확신도: <strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    elif verdict == "likely_real":
        st.markdown(f"""
        <div class="verdict-real">
            <h3>✅ 실제 이미지로 판정</h3>
            <p>확신도: <strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-uncertain">
            <h3>❓ 판정 불확실</h3>
            <p>확신도: <strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 상세 결과
    with st.expander("📋 상세 분석 결과", expanded=True):
        # 판정 근거
        st.subheader("판정 근거")
        reasoning = result.get("reasoning", "")
        for reason in reasoning.split(" | "):
            st.write(f"• {reason}")
        
        st.divider()
        
        # Layer 1: Hash Check (DinoV2)
        st.subheader("Layer 1: Hash Check (DinoV2)")
        hash_result = result.get("hash_result", {})
        col1, col2 = st.columns(2)
        with col1:
            similarity = hash_result.get('similarity', 0)
            st.metric("DinoV2 유사도", f"{similarity:.1%}")
        with col2:
            is_ai = hash_result.get("is_ai", False)
            if is_ai:
                st.error("⚠️ AI 이미지 DB 매칭")
            else:
                st.success("✓ DB 미등록")
        
        st.divider()
        
        # Layer 2: Metadata Analysis
        st.subheader("Layer 2: Metadata Analysis")
        metadata = result.get("metadata_result", {})

        # EXIF 진위성 점수 (새로 추가)
        col1, col2, col3 = st.columns(3)
        with col1:
            exif_score = metadata.get("exif_authenticity_score", 0)
            st.metric("EXIF 진위성", f"{exif_score:.2f}")
            if exif_score >= 0.7:
                st.success("📷 실제 카메라 촬영 가능성")
            elif exif_score >= 0.3:
                st.info("📷 중간 수준")
            else:
                st.warning("⚠️ AI 생성 의심")

        with col2:
            if metadata.get("has_c2pa"):
                st.success("📜 C2PA 존재")
            else:
                st.info("📜 C2PA 없음")

        with col3:
            sig_count = len(metadata.get("ai_tool_signatures", []))
            if sig_count > 0:
                st.error(f"🔍 AI 시그니처: {sig_count}개")
            else:
                st.success("✓ AI 시그니처 없음")

        # EXIF 비정상 패턴 (새로 추가)
        exif_inconsistencies = metadata.get("exif_inconsistencies", [])
        if exif_inconsistencies:
            st.warning("⚠️ **EXIF 비정상 패턴 탐지:**")
            inconsistency_msgs = {
                "editing_software_without_camera": "편집 소프트웨어만 존재 (카메라 정보 없음)",
                "perfect_square_ai_resolution": "AI 생성 특징적 해상도 (512x512, 1024x1024 등)",
                "unrealistic_aperture": "비현실적인 조리개 값",
                "missing_datetime_original": "원본 촬영 시간 정보 누락"
            }
            for inc in exif_inconsistencies:
                st.write(f"  • {inconsistency_msgs.get(inc, inc)}")

        # 상세 정보
        st.markdown("**상세 정보:**")

        if metadata.get("ai_tool_signatures"):
            st.warning(f"🔍 AI 도구: {', '.join(metadata['ai_tool_signatures'])}")

        if metadata.get("software_used"):
            st.info(f"💻 소프트웨어: {metadata['software_used']}")

        if metadata.get("creation_date"):
            st.info(f"📅 촬영/생성 날짜: {metadata['creation_date']}")

        if metadata.get("exif_data"):
            with st.expander("📊 전체 EXIF 데이터 보기"):
                exif_data = metadata["exif_data"]
                # 주요 필드만 먼저 표시
                important_fields = ["Make", "Model", "Software", "DateTime", "DateTimeOriginal",
                                   "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]
                important_data = {k: v for k, v in exif_data.items() if k in important_fields}
                if important_data:
                    st.markdown("**주요 EXIF 정보:**")
                    st.json(important_data)

                st.markdown("**전체 EXIF 데이터:**")
                st.json(exif_data)
        
        st.divider()
        
        # Layer 3: AI Detection
        st.subheader("Layer 3: AI Detection")
        detection = result.get("detection_result")
        if detection:
            st.write(f"**모델**: {detection.get('model_name', 'N/A')}")
            st.write(f"**AI 생성 판정**: {'예' if detection.get('is_ai_generated') else '아니오'}")
            st.write(f"**확신도**: {detection.get('confidence', 0):.1%}")
            
            if detection.get("raw_scores"):
                st.write("**Raw Scores:**")
                for label, score in detection["raw_scores"].items():
                    st.progress(score, text=f"{label}: {score:.1%}")
        else:
            st.info("AI 탐지 스킵됨")
    
    # 실행 시간
    st.caption(f"⏱️ 총 실행 시간: {result.get('total_execution_time_ms', 0):.0f}ms")


if __name__ == "__main__":
    main()
