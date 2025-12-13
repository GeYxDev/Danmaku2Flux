# Danmaku2Flux Project File Structure

**Core Data Processing Layer for Video Recommendation System Based on Danmu Sentiment**

---

## 📁 Project Root Directory (`Danmaku2Flux/`)

This directory contains the complete processing pipeline from raw danmu data to video vector generation, as well as model training and quality evaluation tools.

---

### 🐍 **Core Python Scripts (.py)**

Arranged in data processing pipeline order:

| Filename | Function Description                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`data_preprocessing.py`** | **Data Preprocessing Entry Point**: Cleans raw danmu data (filtering, tokenization), generates structured time-series data, outputs to `processed_danmu_data_115/`                                                                                                                                                                                                                                                                                        |
| **`snownlp_sentiment.py`** | **Sentiment Computation Engine**: Uses SnowNLP library to calculate sentiment scores for each danmu, integrates timestamp and density information, generates complete labeled dataset stored in `vector_danmu_data_115/`                                                                                                                                                                                                                                   |
| **`vector_extractor.py`** | **Vector Extraction Tool**: Extracts key fields (video title, BV ID, 128-dimensional sentiment vector) from JSON file series in `vector_danmu_data_115/`, generates lightweight `simplified_vector_danmu.json` for high-speed reading by transformer model related processing procedures                                                                                                                                                                    |
| **`transformer_recommender.py`** | **Transformer End-to-End Pipeline**: Handles model training, validation, and inference. Input: `simplified_vector_danmu.json`. Output: `transformer_vector_danmu.json` containing complete metadata                                                                                                                                                                                                                                                  |
| **`check_embedding_quality.py`** | **Quality Inspection Suite**:<br>• Basic Similarity Check - Basic similarity statistics<br>• Embedding Diagnostics - Vector distribution diagnostics<br>• Pairwise Cosine Similarity - Pairwise similarity matrix<br>• Nearest Neighbor Similarity - Nearest neighbor quality evaluation<br>• PCA Variance Explained - Principal component variance explanation rate<br>• Running t-SNE - Dynamic clustering visualization, generates `emb_tsne.png` |

---

### 📄 **JSON Data Files (.json)**

| Filename | Function Description                                                                                                                                                                      |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`simplified_vector_danmu.json`** | **Research-Grade Comprehensive Library**: Contains video title, BV ID, 100-dim danmu density vector + 100-dim sentiment vector + metadata for model debugging and analysis                |
| **`transformer_vector_danmu.json`** | **Production-Grade Vector Library**: Contains only `{bv: "...", vector: [], title: "..."}` key-value pairs with minimal size and fastest loading speed for use in recommendation systems |

---

### 📂 **Data Cache Folders**

| Folder Name | Content Description                                                                                                                                                                      |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`danmu_data_115/`** | **Raw Data Repository**: Raw danmu JSON files stored by video episode number, each containing raw timestamps, text content for all danmu of that video                                          |
| **`processed_danmu_data_115/`** | **Preprocessing Cache**: Clean data output from `data_preprocessing.py`, contains tokenized terms and filtered time-series                                                            |
| **`vector_danmu_data_115/`** | **Sentiment Annotation Cache**: The intermediate result generated by `snownlp_dentiment.py`, with emotion values and density features attached to bullet comments in each video fragment |

---

### 🎯 **Models and Configurations**

| Filename | Description |
|----------|-------------|
| **`danmu_transformer_best.pth`** | Transformer model weights with lowest validation loss (recommended for production) |
| **`danmu_transformer_last.pth`** | Last training checkpoint weights (for training resumption)                         |
| **`stopwords.txt`** | Chinese stopwords list, filters meaningless terms during preprocessing                          |
| **`loss_curve.png`** | Training process visualization showing Transformer loss decline trend                          |
| **`emb_tsne.png`** | Vector distribution visualization after t-SNE dimensionality reduction (2D scatter plot)         |

---

### 🛠 Key Dependencies

* **PyTorch (`torch`)**:
The core deep learning framework. It is used in `transformer_recommender.py` to construct, train, and inference the Transformer-based encoder, mapping variable-length sentiment sequences into fixed 128-dimensional embedding vectors.
* **SnowNLP**:
The sentiment analysis engine used in `snownlp_sentiment.py`. It calculates a probabilistic sentiment score (0-1) for every danmu comment, converting unstructured text into the numerical signals required by the model.
* **Jieba**:
Performs precise Chinese text segmentation (tokenization) in `data_preprocessing.py`. It is essential for filtering stop words and cleaning raw danmu text before sentiment analysis.
* **Scikit-learn**:
Powers the quality assurance tools in `check_embedding_quality.py`. It provides algorithms for **PCA** and **t-SNE** (dimensionality reduction) to analyze vector distributions, as well as metrics for cluster analysis.
* **Matplotlib**:
The visualization library used to generate diagnostic charts, including the training loss history (`loss_curve.png`) and the 2D vector embedding scatter plots (`emb_tsne.png`).
* **NumPy & SciPy**:
Handle high-performance numerical computing. **NumPy** manages the dense arrays for time-series data and vector matrices, while **SciPy** is utilized for computing distance metrics (e.g., Cosine Similarity) during quality checks.
* **tqdm**:
Enhances the Command Line Interface (CLI) experience by providing real-time progress bars for long-running tasks such as dataset preprocessing, sentiment batch scoring, and model training epochs.

---

## 🔄 **Standard Running Workflow**

**First-time Initialization:**
```bash
# 1. Clean raw danmu
python data_preprocessing.py

# 2. Calculate sentiment scores and density features
python snownlp_sentiment.py

# 3. Extract simplified vector library
python vector_extractor.py

# 4. Train Transformer model (required for first run)
python transformer_recommender.py --mode train

# 5. Generate complete vector library
python transformer_recommender.py --mode inference

# 6. (Optional) Quality inspection
python check_embedding_quality.py
```

**Subsequent Updates:** Only repeat steps 1-2-3-5 for incremental vector library updates