# 🔍 AI Image Filter

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**3-Layer Verification Service for Filtering AI-Generated Images**

- With the advancement of generative AI, the issue of Data Contamination in training datasets has emerged.
- This service was developed to test whether it is possible to prevent image data contamination using a 3-Layer approach (Hash - Metadata - Open Source Detection Model). It relies on the fact that AI-generated images lack EXIF data (camera model, lens type, shutter speed, GPS location, etc.) found in standard digital photos, and is based on concepts presented in [Provenance Detection for AI-Generated Images: Combining Perceptual Hashing, Homomorphic Encryption, and AI Detection Models](https://arxiv.org/html/2503.11195v1).
- For DinoHash (DinoV2-based perceptual hashing), we used the [ai-vs-human-generated-dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data). We vectorized 39,975 AI-generated images using DinoV2 and saved them as npy files. The similarity threshold was set to 0.85, referencing the DINOv2 paper and medical imaging research, with a gradual score assigned to the uncertain range of 70-85%.
- The Metadata inspection focuses on calculating an EXIF authenticity score, utilizing C2PA Content Credentials verification and AI tool signature detection as auxiliary measures. EXIF analysis calculates an authenticity score (0.0 ~ 1.0) by synthesizing camera information, shooting settings, GPS, etc., and detects abnormal patterns.
- For the AI model, we used HuggingFace's [ai_vs_human_generated_image_detection](https://huggingface.co/dima806/ai_vs_human_generated_image_detection).
- Finally, the results from the Hash, Metadata, and Open Source Detection layers are weighted at 0.3, 0.4, and 0.3 respectively to make a comprehensive judgment.
- Deployed on Hugging Face. [huggingface space](https://huggingface.co/spaces/nepark/ai-image-filter)

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)

---

## ✨ Key Features

### 3-Layer Verification System

| Layer | Function | Description |
|-------|------|------|
| **Layer 1** | Hash Check | AI Image DB Matching based on DinoHash (DinoV2 vector similarity, threshold 0.85), gradual scoring |
| **Layer 2** | Metadata Analysis | EXIF Authenticity Score Calculation (Core) + C2PA Verification/AI Signature Detection (Auxiliary) |
| **Layer 3** | AI Detection | AI Generated Image Classification based on HuggingFace Model |

### Main Characteristics

- 🚀 **Fast Analysis**: Single image analysis completed within 2-5 seconds
- 📦 **Batch Processing**: Simultaneous analysis of up to 50 images
- 📊 **Detailed Report**: Provides analysis results and judgment grounds for each Layer
- 🔌 **REST API**: Scalable API based on FastAPI
- 🎨 **Web UI**: Intuitive interface based on Streamlit

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│                   (streamlit_app.py)                    │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                      │
│                     (app/main.py)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Layer 1   │  │   Layer 2   │  │   Layer 3   │      │
│  │ Hash Check  │  │  Metadata   │  │ AI Detect   │      │
│  │             │  │  Analysis   │  │             │      │
│  │ - DinoV2    │  │ - EXIF      │  │ - HF Model  │      │
│  │   Vector    │  │   Score     │  │ - Inference │      │
│  │ Similarity  │  │ - C2PA/Sign │  │             │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                  Pipeline Service                       │
│           (Comprehensive Verdict + Weighting)           │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/world970511/ai-image-filter.git
cd ai-image-filter
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
cp .env.example .env
# Edit .env file and enter necessary settings
```

### 4. Run Server

```bash
# FastAPI Server (Terminal 1)
uvicorn app.main:app --reload --port 8000

# Streamlit UI (Terminal 2)
streamlit run streamlit_app.py
```

### 5. Access

- **API Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:8501

---

## 📡 API Documentation

### Single Image Analysis

```bash
POST /api/v1/analyze
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@image.jpg"
```

**Response Example:**
```json
{
  "id": "uuid",
  "filename": "image.jpg",
  "final_verdict": "ai_generated",
  "confidence_score": 0.87,
  "reasoning": "🤖 AI Detection Model Verdict: AI Generated (Confidence: 87.0%)",
  "hash_result": { "DinoHash": "..."},
  "metadata_result": { "has_c2pa": false, "ai_tool_signatures": [] },
  "detection_result": { "is_ai_generated": true, "confidence": 0.87 }
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|------|
| POST | `/api/v1/analyze` | Single Image Analysis |
| POST | `/api/v1/analyze/batch` | Batch Analysis (Max 50) |

---

## 🛠 Tech Stack

| Category | Tech |
|------|------|
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Streamlit |
| **AI/ML** | HuggingFace Transformers, PyTorch |
| **Image Processing** | Pillow, imagehash |
| **Deployment** | Docker, HuggingFace Spaces |

---

## 📊 Comparison with Google's SynthID Detector
> Please refer to data-readme.md in the /testIMG folder for a description of the test data images. <br/>
> The test results using testIMG are summarized in [3-Layer Image Filter](https://world970511.github.io/blog/posts/2026-01-19-3-layers-image-filter.html) under the Blog Project category.

---

## License

MIT License - Free to use, modify, and distribute.

---
