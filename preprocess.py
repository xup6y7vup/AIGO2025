import dask.dataframe as dd
import re

# 讀取 CSV，增加 sample 參數以確保足夠取樣
ddf = dd.read_csv('public_x.csv', sample=1000000)

# 建立一個新的欄位清單，只保留符合條件的欄位
keep_columns = []
for col in ddf.columns:
    # 若欄位中包含 "(千)"，直接捨棄
    if '(千)' in col:
        continue
    # 嘗試找出欄位名稱中的「前X天」部分
    m = re.search(r'前(\d+)天', col)
    if m:
        # 如果數字大於 5，則捨棄此欄位
        if int(m.group(1)) > 5:
            continue
    # 其他情況下保留該欄位
    keep_columns.append(col)

# 建立只保留指定欄位的新 DataFrame
ddf_new = ddf[keep_columns]

# 檢查保留的欄位
print("保留的欄位:")
for col in ddf_new.columns:
    print(col)

# 將處理後的資料輸出到單一的 CSV 檔案（需要 Dask 版本支援 single_file）
ddf_new.to_csv('public_x.csv', single_file=True, index=False)

print("前處理完成，結果已儲存到 public_x.csv")
