"""
HD Map异常识别
Logistic Regression基线模型
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sympy.abc import J

df = pd.read_csv("../data/processed/ml_dataset.csv")

print(df.head())
print(df.shape)

road_type_columns = [column for column in df.columns
                     if column.startswith("road_type")]

feature_columns = ["speed_limit","lane_num","road_length_m"]+road_type_columns

X = df[feature_columns]

y = df["is_anomaly"]

X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.3,random_state=42,stratify=y))

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

precision = precision_score(y_test,y_pred,zero_division=0)

recall = recall_score(y_test,y_pred,zero_division=0)

f1 = f1_score(y_test,y_pred,zero_division=0)

print("Aaccuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)
print("\nClassification Report:")
print(classification_report(y_test,y_pred,zero_division=0))