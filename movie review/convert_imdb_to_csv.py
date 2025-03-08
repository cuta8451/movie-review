import os
import pandas as pd

# 設定資料夾路徑
data_dir = "C:\\Users\\ASUS\\Downloads\\aclImdb"  # 
sets = ["train", "test"]  # 訓練集和測試集
categories = ["pos", "neg"]  # 影評的情感標籤

# 建立一個空的列表來存放所有影評
data = []

# 遍歷資料夾
for data_set in sets:
    for category in categories:
        folder_path = os.path.join(data_dir, data_set, category)  # 建立完整的資料夾路徑
        
        # 確保資料夾存在
        if not os.path.exists(folder_path):
            print(f"❌ 資料夾 {folder_path} 不存在，跳過...")
            continue
        
        # 遍歷該資料夾內的所有 txt 文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # 讀取文本內容
            with open(file_path, "r", encoding="utf-8") as file:
                review = file.read().strip()  # 去除前後空白
                sentiment = "positive" if category == "pos" else "negative"  # 標註情感
                
                # 加入到數據列表
                data.append([review, sentiment])

# 轉為 DataFrame
df = pd.DataFrame(data, columns=["review", "sentiment"])

# 儲存為 CSV
csv_file = "imdb_reviews.csv"
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"✅ 資料已成功彙整至 {csv_file}，共 {len(df)} 條影評！")
