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

# ============ Configuration ============
API_URL = "http://localhost:8000/api/v1"  # FastAPI Server Address

# Page Configuration
st.set_page_config(
    page_title="AI Image Filter Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ CSS Styles ============
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.markdown(
        '<p class="main-header">🔍 AI Image Filter Pipeline</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">3-Layer Verification System for Filtering AI-Generated Images from ML Training Datasets</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        api_url = st.text_input("API URL", value=API_URL)

        st.divider()

        st.header("📊 Analysis Pipeline")
        st.markdown("""
        **Layer 1**: DinoV2 Hash Check
        - facebook/dinov2-small vector similarity
        
        **Layer 2**: Metadata Analysis
        - EXIF Authenticity Score
        - EXIF Abnormal Pattern Detection
        - C2PA Content Credentials
        - AI Tool Signatures

        **Layer 3**: AI Detection
        - [HuggingFace Model](https://huggingface.co/dima806/ai_vs_human_generated_image_detection)
        """)
        st.info(
            "ℹ️ Stateless Mode - Does not use a database. All analyses are processed in real-time."
        )

        st.divider()

    # Main Tabs
    tab1, tab2 = st.tabs(["📤 Single Image Analysis", "📦 Batch Analysis"])

    # ============ Tab 1: Single Image Analysis ============
    with tab1:
        st.header("Single Image Analysis")

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            key="single_upload",
        )

        if uploaded_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                st.caption(
                    f"Filename: {uploaded_file.name} | Size: {uploaded_file.size:,} bytes"
                )

            with col2:
                st.subheader("🔬 Analysis Results")

                if st.button("🚀 Start Analysis", type="primary", key="analyze_single"):
                    with st.spinner("Analyzing..."):
                        try:
                            # Call API
                            uploaded_file.seek(0)
                            files = {
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type,
                                )
                            }

                            response = requests.post(
                                f"{api_url}/analyze", files=files, timeout=60
                            )

                            if response.status_code == 200:
                                result = response.json()
                                display_result(result)
                            else:
                                st.error(f"Analysis Failed: {response.text}")

                        except requests.exceptions.ConnectionError:
                            st.error(
                                "⚠️ Cannot connect to API server. Please check if the server is running."
                            )
                            st.info("Local Test: `uvicorn app.main:app --reload`")
                        except Exception as e:
                            st.error(f"Error Occurred: {e}")

    # ============ Tab 2: Batch Analysis ============
    with tab2:
        st.header("Batch Image Analysis")
        st.info("You can analyze up to 50 images at once.")

        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} files selected")

            if st.button(
                "🚀 Start Batch Analysis", type="primary", key="analyze_batch"
            ):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, file in enumerate(uploaded_files):
                    status_text.text(
                        f"Analyzing: {file.name} ({i + 1}/{len(uploaded_files)})"
                    )

                    try:
                        file.seek(0)
                        files = {"file": (file.name, file.getvalue(), file.type)}

                        response = requests.post(
                            f"{api_url}/analyze", files=files, timeout=60
                        )

                        if response.status_code == 200:
                            result = response.json()
                            metadata = result.get("metadata_result", {})
                            hash_res = result.get("hash_result", {})
                            results.append(
                                {
                                    "Filename": file.name,
                                    "Verdict": result.get("final_verdict", "unknown"),
                                    "Confidence": f"{result.get('confidence_score', 0):.1%}",
                                    "DinoV2 Similarity": f"{hash_res.get('similarity', 0):.1%}",
                                    "EXIF Authenticity": f"{metadata.get('exif_authenticity_score', 0):.2f}",
                                    "AI Signatures": ", ".join(
                                        metadata.get("ai_tool_signatures", [])
                                    )
                                    or "-",
                                    "EXIF Abnormalities": len(
                                        metadata.get("exif_inconsistencies", [])
                                    ),
                                }
                            )
                        else:
                            results.append(
                                {
                                    "Filename": file.name,
                                    "Verdict": "error",
                                    "Confidence": "-",
                                    "DinoV2 Similarity": "-",
                                    "EXIF Authenticity": "-",
                                    "AI Signatures": "-",
                                    "EXIF Abnormalities": "-",
                                }
                            )
                    except Exception as e:
                        results.append(
                            {
                                "Filename": file.name,
                                "Verdict": "error",
                                "Confidence": "-",
                                "DinoV2 Similarity": "-",
                                "EXIF Authenticity": "-",
                                "AI Signatures": str(e)[:30],
                                "EXIF Abnormalities": "-",
                            }
                        )

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("✅ Analysis Complete!")

                # Result Table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Statistics
                col1, col2, col3 = st.columns(3)
                ai_count = sum(1 for r in results if r["Verdict"] == "ai_generated")
                real_count = sum(1 for r in results if r["Verdict"] == "likely_real")
                uncertain_count = sum(1 for r in results if r["Verdict"] == "uncertain")

                col1.metric("🤖 AI Generated", ai_count)
                col2.metric("✅ Real Image", real_count)
                col3.metric("❓ Uncertain", uncertain_count)

                # CSV Download
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    f"ai_filter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                )


def display_result(result: dict):
    """Display Analysis Results"""
    verdict = result.get("final_verdict", "unknown")
    confidence = result.get("confidence_score", 0)

    # Display Verdict
    if verdict == "ai_generated":
        st.markdown(
            f"""
        <div class="verdict-ai">
            <h3>🤖 Verdict: AI Generated Image</h3>
            <p>Confidence: <strong>{confidence:.1%}</strong></p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif verdict == "likely_real":
        st.markdown(
            f"""
        <div class="verdict-real">
            <h3>✅ Verdict: Likely Real Image</h3>
            <p>Confidence: <strong>{confidence:.1%}</strong></p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="verdict-uncertain">
            <h3>❓ Verdict: Uncertain</h3>
            <p>Confidence: <strong>{confidence:.1%}</strong></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Detailed Results
    with st.expander("📋 Detailed Analysis Results", expanded=True):
        # Verdict Reasoning
        st.subheader("Reasoning")
        reasoning = result.get("reasoning", "")
        # Note: If the backend returns reasoning in Korean, this split/display might still show Korean.
        # We should probably update the backend to return English reasoning as well.
        for reason in reasoning.split(" | "):
            st.write(f"• {reason}")

        st.divider()

        # Layer 1: Hash Check (DinoV2)
        st.subheader("Layer 1: Hash Check (DinoV2)")
        hash_result = result.get("hash_result", {})
        col1, col2 = st.columns(2)
        with col1:
            similarity = hash_result.get("similarity", 0)
            st.metric("DinoV2 Similarity", f"{similarity:.1%}")
        with col2:
            is_ai = hash_result.get("is_ai", False)
            if is_ai:
                st.error("⚠️ Matched in AI Image DB")
            else:
                st.success("✓ Not found in DB")

        st.divider()

        # Layer 2: Metadata Analysis
        st.subheader("Layer 2: Metadata Analysis")
        metadata = result.get("metadata_result", {})

        # EXIF Authenticity Score
        col1, col2, col3 = st.columns(3)
        with col1:
            exif_score = metadata.get("exif_authenticity_score", 0)
            st.metric("EXIF Authenticity", f"{exif_score:.2f}")
            if exif_score >= 0.7:
                st.success("📷 High likelihood of real camera")
            elif exif_score >= 0.3:
                st.info("📷 Medium likelihood")
            else:
                st.warning("⚠️ Suspected AI Generation")

        with col2:
            if metadata.get("has_c2pa"):
                st.success("📜 C2PA Present")
            else:
                st.info("📜 C2PA Absent")

        with col3:
            sig_count = len(metadata.get("ai_tool_signatures", []))
            if sig_count > 0:
                st.error(f"🔍 AI Signatures: {sig_count}")
            else:
                st.success("✓ No AI Signatures")

        # EXIF Abnormal Patterns
        exif_inconsistencies = metadata.get("exif_inconsistencies", [])
        if exif_inconsistencies:
            st.warning("⚠️ **EXIF Abnormal Patterns Detected:**")
            inconsistency_msgs = {
                "editing_software_without_camera": "Editing software only (No camera info)",
                "perfect_square_ai_resolution": "Characteristic AI resolution (512x512, 1024x1024, etc.)",
                "unrealistic_aperture": "Unrealistic aperture value",
                "missing_datetime_original": "Missing original capture time",
            }
            for inc in exif_inconsistencies:
                st.write(f"  • {inconsistency_msgs.get(inc, inc)}")

        # Detailed Info
        st.markdown("**Detailed Info:**")

        if metadata.get("ai_tool_signatures"):
            st.warning(f"🔍 AI Tools: {', '.join(metadata['ai_tool_signatures'])}")

        if metadata.get("software_used"):
            st.info(f"💻 Software: {metadata['software_used']}")

        if metadata.get("creation_date"):
            st.info(f"📅 Capture/Creation Date: {metadata['creation_date']}")

        if metadata.get("exif_data"):
            with st.expander("📊 View Full EXIF Data"):
                exif_data = metadata["exif_data"]
                # Display major fields first
                important_fields = [
                    "Make",
                    "Model",
                    "Software",
                    "DateTime",
                    "DateTimeOriginal",
                    "ExposureTime",
                    "FNumber",
                    "ISOSpeedRatings",
                    "FocalLength",
                ]
                important_data = {
                    k: v for k, v in exif_data.items() if k in important_fields
                }
                if important_data:
                    st.markdown("**Major EXIF Info:**")
                    st.json(important_data)

                st.markdown("**Full EXIF Data:**")
                st.json(exif_data)

        st.divider()

        # Layer 3: AI Detection
        st.subheader("Layer 3: AI Detection")
        detection = result.get("detection_result")
        if detection:
            st.write(f"**Model**: {detection.get('model_name', 'N/A')}")
            st.write(
                f"**AI Generated Verdict**: {'Yes' if detection.get('is_ai_generated') else 'No'}"
            )
            st.write(f"**Confidence**: {detection.get('confidence', 0):.1%}")

            if detection.get("raw_scores"):
                st.write("**Raw Scores:**")
                for label, score in detection["raw_scores"].items():
                    st.progress(score, text=f"{label}: {score:.1%}")
        else:
            st.info("AI Detection was not performed.")

    # Execution Time
    st.caption(
        f"⏱️ Total Execution Time: {result.get('total_execution_time_ms', 0):.0f}ms"
    )


if __name__ == "__main__":
    main()
