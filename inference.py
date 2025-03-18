import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 定義與訓練時相同的 MLP 模型架構
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 讀取公測資料，檔案中必須包含 "ID" 欄位
    data = pd.read_csv('public_x_preprocess.csv')
    
    # 取得 ID 欄位（將在結果中保留）
    ids = data['ID']
    # 特徵資料：排除 ID 欄位，確保轉為數值型
    X = data.drop(['ID'], axis=1).values.astype(np.float32)
    
    # 轉換為 torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # 建立模型：根據特徵數決定輸入維度
    input_dim = X_tensor.shape[1]
    model = MLP(input_dim).to(device)
    
    # 載入先前儲存的模型參數 (請確認 mlp_model.pth 路徑正確)
    model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
    model.eval()
    
    # 預測
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        # 使用 0.5 閥值轉換預測結果為 0 或 1
        predictions = (outputs > 0.5).cpu().numpy().astype(int)
    
    # 組成結果 DataFrame
    submission = pd.DataFrame({
        "ID": ids,
        "飆股": predictions
    })
    
    # 儲存結果至 CSV，符合上傳格式要求：
    # UTF-8 (無 BOM)、Unix 換行符號、第一列為欄位名稱 "ID","飆股"
    submission.to_csv("submission.csv", index=False, encoding="utf-8", lineterminator="\n")
    print("預測結果已儲存到 submission.csv")
    
if __name__ == "__main__":
    main()
