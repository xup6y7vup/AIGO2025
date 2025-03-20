import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, AdamW
import torch.nn as nn
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

# 假設你的原始資料已讀入 df 中
print("匯入資料開始")
with ProgressBar():
    ddf = dd.read_csv('features_only_v11.csv', sample=1000000)
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


# 將資料分為 train 和 test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['ID', '飆股'], axis=1), df['飆股'], test_size=0.2, stratify=df['飆股'], random_state=42
)

# 使用 SMOTE 進行平衡
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 轉換成 TensorDataset
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 使用 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion = FocalLoss(alpha=pos_weight, gamma=2)

# 使用 Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch, seq=1, model_dim)
        x = self.transformer(x).squeeze(1)
        return self.fc(x)

model = TransformerClassifier(input_dim=X_train_tensor.shape[1]).to(device)

criterion = FocalLoss(alpha=135, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 訓練模型並儲存最佳模型 (自動適應 input_dim)
from sklearn.metrics import classification_report, f1_score

input_dim = X_train_tensor.shape[1]
model = TransformerClassifier(input_dim=input_dim).to(device)
#model.load_state_dict(torch.load("best_transformer_model.pth", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_f1 = 0
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_preds = []
    train_labels = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb).squeeze()
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_preds.extend((torch.sigmoid(outputs) >= 0.5).cpu().numpy())
        train_labels.extend(yb.cpu().numpy())

    train_f1 = f1_score(train_labels, train_preds)

    # 評估階段
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        preds = (torch.sigmoid(test_outputs) >= 0.5).cpu().numpy()
        test_f1 = f1_score(y_test, preds)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Train F1-score: {train_f1:.4f}, Test F1-score: {test_f1:.4f}")

    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), "best_transformer_model.pth")
        print(f"New best model saved with Test F1-score: {best_f1:.4f}")