import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 下載 NLTK 資源
nltk.download('stopwords')
nltk.download('punkt')

# 讀取 IMDb 影評數據 (請先下載)
df = pd.read_csv("imdb-dataset.csv")

# 只保留 影評文本 (review) 和 標籤 (sentiment)
df = df[['review', 'sentiment']]

# 將標籤轉換為數字 (positive -> 1, negative -> 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# NLP 預處理函數
def clean_text(text):
    text = text.lower()  # 轉換小寫
    text = re.sub(r'<.*?>', '', text)  # 去除 HTML 標籤
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除標點符號與數字
    words = word_tokenize(text)  # 斷詞
    words = [word for word in words if word not in stopwords.words('english')]  # 去除停用詞
    return " ".join(words)

# 應用 NLP 預處理
df['clean_review'] = df['review'].apply(clean_text)

# 轉換文本為向量 (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練 Logistic Regression 模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 測試並輸出準確度
y_pred = model.predict(X_test)
print("模型準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 儲存模型
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 儲存 TF-IDF 向量化器
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
