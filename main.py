import dask.dataframe as dd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from dask.diagnostics import ProgressBar
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

print("匯入資料開始")
with ProgressBar():
    ddf = dd.read_csv('training_processed_v1.csv', sample=1000000)
    df = ddf.compute()

print("原始資料形狀:", df.shape)
# 取得除 "ID" 外所有欄位
cols = df.columns.drop('ID')
df[cols] = df[cols].replace({'%': '', ',': ''}, regex=True)
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
print("轉換後資料形狀:", df.shape)

# 計算 NaN 與 0 的數量
nan_count = df[cols].isna().sum().sum()
zero_count = (df[cols] == 0).sum().sum()
print("NaN的總數:", nan_count)
print("0的總數:", zero_count)

# 填補 NaN
df[cols] = df[cols].fillna(-1)
print("填補 NaN 後資料形狀:", df.shape)
print(df.head())
print("匯入資料結束")

print("切分資料集開始")
# 分離特徵與目標變數 (排除 "ID" 與 "飆股")
X = df.drop(['ID', '飆股'], axis=1).values.astype(np.float32)
y = df['飆股'].values.astype(np.float32)
print("y shape:", y.shape)
print("0的數量:", np.sum(y == 0))
print("1的數量:", np.sum(y == 1))

# 使用 stratified split 保持類別比例
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("切分資料集結束")

print("執行 SMOTE 過抽樣...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("SMOTE 完成，訓練集大小:", X_train.shape, "1的數量:", np.sum(y_train == 1))

print("轉換資料集開始")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val   = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val   = torch.tensor(y_val, dtype=torch.float32).to(device)
print("轉換資料集結束")

# ------------------------------
# 定義更強大的模型：DeepMLP (含 BatchNorm 與 Dropout)
# ------------------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 64], dropout_rate=0.5):
        """
        使用兩層隱藏層，仍保留 BatchNorm、ReLU 與 Dropout 的結構。
        hidden_dims: 你可以根據需要調整隱藏層大小，這裡示範用兩層。
        """
        super(DeepMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        # 最後一層輸出 logits (未經 sigmoid)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = DeepMLP(input_dim).to(device)

# ------------------------------
# 定義 Focal Loss 損失函數
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        alpha: 正類權重 (依據正負比例設定)
        gamma: 調整因子，通常設為 2
        reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # 預測正確的機率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 計算全資料中正負比例 (用原始資料作參考)
total_zeros = np.sum(y == 0)
total_ones = np.sum(y == 1)
pos_weight = total_zeros / total_ones
print("原始資料 pos_weight =", pos_weight)

# 使用 Focal Loss
criterion = FocalLoss(alpha=pos_weight, gamma=2, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("訓練模型開始")
epochs = 1000
best_val_f1 = 0.0  # 追蹤最佳驗證 F1-score
for epoch in range(epochs):
    # -------------------------
    # 訓練階段
    # -------------------------
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()  # 輸出 logits
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        probs_train = torch.sigmoid(outputs)
        predicted_train = (probs_train > 0.5).float()
        train_acc = (predicted_train == y_train).float().mean().item()
        train_f1 = f1_score(y_train.cpu().numpy(), predicted_train.cpu().numpy())
        train_precision = precision_score(y_train.cpu().numpy(), predicted_train.cpu().numpy())
        train_recall = recall_score(y_train.cpu().numpy(), predicted_train.cpu().numpy())
        train_cm = confusion_matrix(y_train.cpu().numpy(), predicted_train.cpu().numpy())
    
    # -------------------------
    # 驗證階段
    # -------------------------
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()  # 輸出 logits
        probs_val = torch.sigmoid(val_outputs)
        predicted_val = (probs_val > 0.5).float()
        val_loss = criterion(val_outputs, y_val)
        val_acc = (predicted_val == y_val).float().mean().item()
        val_f1 = f1_score(y_val.cpu().numpy(), predicted_val.cpu().numpy())
        val_precision = precision_score(y_val.cpu().numpy(), predicted_val.cpu().numpy())
        val_recall = recall_score(y_val.cpu().numpy(), predicted_val.cpu().numpy())
        val_cm = confusion_matrix(y_val.cpu().numpy(), predicted_val.cpu().numpy())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {loss.item():.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Train Confusion Matrix:\n{train_cm}")
        print(f"  Val   Loss: {val_loss.item():.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"  Val   Confusion Matrix:\n{val_cm}")
    
    # 若驗證 F1 超過目前最佳，則儲存模型 (僅儲存最佳模型)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "deepmlp_model.pth")
        print(f"模型已儲存，當前最佳驗證 F1: {best_val_f1:.4f}")

print(f"最佳驗證 F1: {best_val_f1:.4f}")
print("訓練模型結束")
