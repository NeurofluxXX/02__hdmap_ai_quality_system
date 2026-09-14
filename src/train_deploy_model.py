import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../data/processed/ml_training_dataset.csv")
feature_columns = ["speed_limit","lane_num","road_length_m","road_type"]
X = df[feature_columns]
y = df["is_anomaly"]

numeric_features = ["speed_limit","lane_num","road_length_m"]
categorical_features = ["road_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("road_type_encoder",OneHotEncoder(handle_unknown="ignore"),categorical_features)],
    remainder = "passthrough")

model_pipeline = Pipeline(steps=[("preprocessor",preprocessor),("classifier",RandomForestClassifier(n_estimators=100,random_state=42))])

model_pipeline.fit(X,y)

joblib.dump(model_pipeline,"../models/hdmap_rf_pipeline.joblib")
print("模型保存成功：""../models/hdmap_rf_pipeline.joblib")
