import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv("../data/processed/ml_training_dataset.csv")
df = pd.get_dummies(df, columns=["road_type"],dtype=int)

road_type_columns = [column for column in df.columns if column.startswith("road_type_")]
feature_columns = ["speed_limit","lane_num","road_length_m"] + road_type_columns
X = df[feature_columns]
y = df["is_anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self,x):
        return self.network(x)

input_dim = X_train_tensor.shape[1]
model = MLP(input_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
loss_history = []

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_X).squeeze()

        loss = criterion(outputs,batch_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)


    if (epoch +1) % 10 ==0:
        print(f"Epoch{epoch +1},"
              f"Average Loss:{average_loss:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test_tensor).squeeze()
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).int()

y_true = y_test_tensor.int().numpy()
y_pred = predictions.numpy()

mlp_cm = confusion_matrix(y_true,y_pred)
print("\nMLP Confusion Matrix:")
print(mlp_cm)

print("Accuracy:",accuracy_score(y_true,y_pred))
print("Precision:",precision_score(y_true,y_pred))
print("Recall:",recall_score(y_true,y_pred))
print("F1:",f1_score(y_true,y_pred))
print("Classification Report:",classification_report(y_true,y_pred))

accuracy = accuracy_score(y_true,y_pred)
precision = precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)

mlp_result = {"model":"MLP",
              "accuracy":accuracy,
              "precision":precision,
              "recall":recall,
              "f1":f1}

mlp_result_df = pd.DataFrame([mlp_result])
mlp_result_df.to_csv("../outputs/mlp_result.csv",index=False)

epochs_list = list(range(1,epochs +1))
loss_df = pd.DataFrame({"epoch":epochs_list,"loss":loss_history})
loss_df.to_csv("../outputs/mlp_loss_history.csv",index=False)