import dask.dataframe as dd

# 指定 sample 為 1000000 bytes（你也可以根據實際情況調整這個數值）
df = dd.read_csv('training.csv', sample=1000000)

# 查看所有欄位
print("CSV 欄位名稱：")
print(df.columns)

# 取得所有欄位名稱，轉換為 list
columns = df.columns.tolist()

# 輸出欄位名稱至文字檔，每一行一個欄位
with open('columns.txt', 'w', encoding='utf-8') as f:
    for col in columns:
        f.write(str(col) + '\n')

print("欄位名稱已成功儲存到 columns.txt")

# 取得第一筆資料
first_row = df.head(1)  # 這會回傳一個 pandas DataFrame
print("\n第一筆資料：")
print(first_row)
