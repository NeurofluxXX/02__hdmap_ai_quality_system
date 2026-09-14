"""
HD Map异常识别
Random Forest模型
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("../data/processed/ml_training_dataset.csv")

df = pd.get_dummies(df,columns=["road_type"],dtype=int)

feature_columns = [column for column in df.columns
                   if column.startswith("road_type")]

feature_columns +=["speed_limit","lane_num","road_length_m"]

X = df[feature_columns]
y = df["is_anomaly"]
print("选中的特征列表：", feature_columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

importance = pd.DataFrame({
    "feature":X.columns,
    "importance":model.feature_importances_
})

importance = importance.sort_values("importance",ascending=False)
print(importance)


