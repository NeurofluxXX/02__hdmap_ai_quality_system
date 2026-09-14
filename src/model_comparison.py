import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("../data/processed/ml_training_dataset.csv")
df = pd.get_dummies(df,columns=["road_type"],dtype=int)

road_type_columns = [column for column in df.columns if column.startswith("road_type")]
feature_columns = ["speed_limit","lane_num","road_length_m"] + road_type_columns

X = df[feature_columns]
y = df["is_anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model_name, y_true, y_pred):
    return {"model":model_name,
            "accuracy":accuracy_score(y_true,y_pred),
            "precision":precision_score(y_true,y_pred),
            "recall":recall_score(y_true,y_pred),
            "f1":f1_score(y_true,y_pred)}

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled,y_train)
logistic_pred = logistic_model.predict(X_test_scaled)
logistic_result = evaluate_model("Logistic Regression",y_test,logistic_pred)

rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)  ##
rf_result = evaluate_model("Random Forest",y_test,rf_pred)
rf_cm = confusion_matrix(y_test,rf_pred)

rf_importance = pd.DataFrame({"feature":X.columns,"importance":rf_model.feature_importances_})
rf_importance = (rf_importance.sort_values("importance",ascending=False))
rf_importance.to_csv("../outputs/rf_feature_importance.csv",index=False)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)


xgb_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
xgb_model.fit(X_train,y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_result = evaluate_model("XGBoost",y_test,xgb_pred)

results = [logistic_result,rf_result,xgb_result]
results_df = pd.DataFrame(results)

print("\n========== Model Comparison ==========")
print(results_df)
results_df.to_csv("../outputs/model_comparison.csv",index=False)
