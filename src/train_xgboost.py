import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("XGBoost version=",xgboost.__version__)

df = pd.read_csv("../data/processed/ml_training_dataset.csv")

df = pd.get_dummies(df,columns=["road_type"],dtype=int)

feature_columns = [column for column in df.columns if column.startswith("road_type")]

feature_columns +=["speed_limit","lane_num","road_length_m"]

X = df[feature_columns]
y = df["is_anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model.fit(X_train,y_train) # 模型训练
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

result = {}
result["model"] = "XGBoost"
result["accuracy"] = accuracy_score(y_test,y_pred)



