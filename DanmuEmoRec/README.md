# DanmuEmoRec

**Bilibili Video Recommendation System Based on Danmu Sentiment**

---

## Project Overview

DanmuEmoRec is a recommendation system that analyzes video sentiment through danmu (bullet comments). When users watch videos on Bilibili, the extension automatically analyzes the sentiment variation curve of the current video and recommends 5 videos with similar emotional patterns, enabling a personalized "recommend by sentiment" experience.

---

## Features

- 🎯 **Sentiment Analysis**: Analyzes video sentiment changes through danmu text, generating 128-dimensional sentiment vectors
- 🔌 **Browser Extension**: One-click recommendation trigger, seamlessly integrated into the Bilibili viewing experience
- 🤖 **Intelligent Matching**: Calculates video similarity based on Transformer models for precise recommendations
- ⚡ **Real-time Response**: Average time from click to recommendation < 5 seconds

---

## Project Structure

```
DanmuEmoRec/
├── backend/                 # Python Backend Service
│   ├── app.py              # Flask Main Application
│   ├── danmu_fetcher.py    # Danmu Fetcher Module
│   ├── sentiment_analyzer.py # Sentiment Analyzer Module
│   ├── transformer_model.py # Vector Conversion Model
│   ├── recommender.py      # Recommendation Algorithm
│   └── requirements.txt    # Python Dependencies
│
└── extension/              # Chrome Extension
    ├── manifest.json       # Extension Configuration
    ├── popup.html          # Popup UI
    ├── popup.js            # Extension Logic
    ├── content.js          # Content Script
    └── icons/              # Extension Icons
```

---

## Tech Stack

- **Backend**: Python 3.8+ / Flask / Transformers / scikit-learn
- **Extension**: JavaScript / Chrome Extension API
- **Data**: Bilibili Danmu API / Self-built Video Vector Database

---

## Installation & Deployment

### Backend Deployment

```bash
cd backend
pip install -r requirements.txt
python app.py
# Service will run on http://localhost:5000
```

### Extension Installation

1. Open Chrome browser and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top-right corner
3. Click "Load unpacked" and select the `extension` folder

---

## Usage

1. **Watch a video**: Play any video on Bilibili
2. **Click Recommend**: Click the extension icon in the top-right corner of the browser
3. **Wait for analysis**: The extension automatically extracts the BV ID and sends it to the backend
4. **View results**: See the 5 sentiment-similar video recommendations in the popup window
5. **Click to watch**: Click video links to open them in new tabs

---

## API

### Get Recommended Videos

- **URL**: `POST /api/recommend`
- **Parameters**: `{"bv": "BV1xx411c7mD"}`
- **Response**: 
```json
{
  "status": "success",
  "current_video": {...},
  "recommendations": [
    {"title": "...", "bv": "BV1...", "similarity": 0.87}
  ]
}
```

---

## Development Notes

### Environment Requirements

- Python 3.8+
- Chrome 88+

---

## Notes

- ⚠️ **API Rate Limit**: Bilibili Danmu API has access frequency limits. It is recommended to add request delays
- 🔑 **Data Privacy**: The extension only extracts the video BV ID and does not collect personal user information
- 📄 **Copyright Notice**: This project is for educational and research purposes only. Please comply with Bilibili's Terms of Service