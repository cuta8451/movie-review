# 🎬 Movie Review Sentiment Analysis

## 📌 Project Overview
This project aims to analyze the sentiment of movie reviews using **Natural Language Processing (NLP) and Machine Learning models**. The goal is to classify reviews as either **positive** or **negative** based on the provided dataset.

## 🎯 Goals
- Preprocess and clean movie reviews from **IMDb dataset**.
- Train a **Machine Learning model** (Logistic Regression / Random Forest / LSTM / BERT) to classify reviews.
- Deploy a **Flask API** to serve predictions.
- Optionally, create a **Streamlit web app** for user interaction.

## 📂 Dataset
- **IMDb Large Movie Review Dataset** ([Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/))
- Contains **50,000** movie reviews labeled as **positive** or **negative**.
- Split into **25,000 training** and **25,000 testing** samples.
- Folder structure:
  ```
  aclImdb/
  │── train/
  │   │── pos/  (positive reviews)
  │   │── neg/  (negative reviews)
  │── test/
  │   │── pos/  (positive reviews)
  │   │── neg/  (negative reviews)
  ```

## 🛠 Tech Stack
- **Programming Language**: Python 🐍
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - NLP: `nltk`, `re`
  - Machine Learning: `sklearn`, `tensorflow` (for deep learning models)
  - Web Deployment: `Flask`, `Streamlit`

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://gitlab.com/your_repo/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Process Dataset & Train Model
```sh
python train_model.py
```

### 4️⃣ Run Flask API
```sh
python app.py
```
Access API at: `http://127.0.0.1:5000/predict`

### 5️⃣ Run Streamlit Web App (Optional)
```sh
streamlit run streamlit_app.py
```

## 🚀 API Usage (Flask)
Send a `POST` request to `/predict` endpoint with a movie review:
```sh
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"review": "This movie is amazing!"}'
```
Response:
```json
{
  "review": "This movie is amazing!",
  "sentiment": "positive"
}
```

## 📈 Future Improvements
- Implement **BERT / Transformer** for better accuracy.
- Deploy on **Google Cloud / AWS**.
- Build a **LINE Bot** to analyze movie reviews in real-time.

---

💡 **Developed by [Your Name]** | 🏗 Open for contributions!

