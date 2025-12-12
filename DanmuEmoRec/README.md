# DanmuEmoRec

**Bilibili Video Recommendation System Based on Danmu Sentiment**

---

## Project Overview

DanmuEmoRec is a recommendation system that analyzes video sentiment through danmu (bullet comments). When users watch videos on Bilibili, the extension automatically analyzes the sentiment variation curve of the current video and recommends 5 videos with similar emotional patterns, enabling a personalized "recommend by sentiment" experience.

---

## Features

- 🎯 **Sentiment Analysis**: Analyzes video sentiment changes through danmu text using SnowNLP and Transformer models.
- 🔌 **Browser Extension**: One-click recommendation trigger, seamlessly integrated into the Bilibili viewing experience.
- 🤖 **Intelligent Matching**: Uses a specialized Transformer model (`danmu_transformer_best.pth`) to generate 128-dimensional sentiment vectors.
- ⚡ **Real-time Response**: Fast retrieval using a pre-computed vector database (`transformer_vector_danmu.json`).

---

## Project Structure

```text
DanmuEmoRec/
├── backend/                          # Python Backend Service
│   ├── core/                         # Core Logic Modules
│   │   ├── __init__.py
│   │   ├── database.py               # Vector Database Management
│   │   ├── pipeline.py               # Recommendation Pipeline Logic
│   │   └── services.py               # Crawler, Preprocessing & Model Services
│   ├── app.py                        # FastAPI Main Application
│   ├── danmu_transformer_best.pth    # Trained Transformer Model Weights
│   ├── requirements.txt              # Python Dependencies
│   └── transformer_vector_danmu.json # Pre-computed Video Vector Database
│
└── extension/                        # Chrome Extension
    ├── icons/                        # Extension Icons
    ├── manifest.json                 # Extension Configuration
    ├── popup.html                    # Popup UI
    └── popup.js                      # Extension Logic
```

---

## Tech Stack

- **Backend**: Python 3.8+ / **FastAPI** / **Uvicorn**
- **AI & Data**: **PyTorch** / Transformers / scikit-learn / SnowNLP / NumPy
- **Extension**: JavaScript / HTML / CSS
- **Data Source**: Bilibili Danmu XML / Local JSON Vector Database

---

## Installation & Deployment

### Backend Deployment

1.  **Install Dependencies**:

    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2.  **Check Resources**:
    Ensure `danmu_transformer_best.pth` and `transformer_vector_danmu.json` are present in the `backend` directory.

3.  **Run Server**:

    ```bash
    python app.py
    # OR using uvicorn directly:
    # uvicorn app:app --reload --port 8000
    ```

    *Service will run on http://127.0.0.1:8000*

### Extension Installation

1.  Open Chrome browser and navigate to `chrome://extensions/`.
2.  Enable **"Developer mode"** in the top-right corner.
3.  Click **"Load unpacked"** and select the `extension` folder from this project.

---

## Usage

1.  **Watch a video**: Play any video on Bilibili.
2.  **Click Recommend**: Click the extension icon in the top-right corner of the browser.
3.  **Wait for analysis**: The extension extracts the BV ID and requests recommendations from the local backend.
4.  **View results**: See the sentiment-similar video recommendations in the popup window.

---

## API

### Get Recommended Videos

- **URL**: `GET /recommend`

- **Parameters**: `?bv=BV1xx411c7mD`

- **Example Request**:
  `http://127.0.0.1:8000/recommend?bv=BV1xx411c7mD`

- **Response Example**:

```json
{
  "code": 200,
  "status": "success",
  "data": [
    {
      "title": "Video Title Example",
      "bv": "BV1vz4y1...",
      "link": "https://www.bilibili.com/video/BV1vz4y1...",
      "score": 0.95
    }
    // ... more results
  ]
}
```

---

## Development Notes

### Environment Requirements

- Python 3.8+
- Chrome 88+
- (Optional) CUDA-enabled GPU for faster model inference

---

## Notes

- ⚠️ **API Rate Limit**: The crawler component interacts with Bilibili's public interfaces. Use responsibly.
- 🔑 **Local Database**: The system relies on `transformer_vector_danmu.json` for fast lookups. If a video is not in the DB, it will trigger real-time crawling and inference (slower).
- 📄 **Copyright Notice**: This project is for educational and research purposes only.