import dask.dataframe as dd

# 讀取要保留的欄位名稱，假設每行一個欄位名稱
with open("columns_processed_v1.txt", "r", encoding="utf-8") as f:
    keep_columns = [line.strip() for line in f if line.strip()]

print("要保留的欄位:")
for col in keep_columns:
    print(col)

# 讀取原始 CSV，設定 sample 大小以確保取樣充足
ddf = dd.read_csv('training_processed.csv', sample=1000000)

# 只保留指定的欄位
ddf_new = ddf[keep_columns]

# 檢查保留的欄位
print("篩選後的欄位:")
for col in ddf_new.columns:
    print(col)

# 將處理後的資料輸出到單一的 CSV 檔案
ddf_new.to_csv('training_processed_v1.csv', single_file=True, index=False)
print("前處理完成，結果已儲存到 training_processed.csv")