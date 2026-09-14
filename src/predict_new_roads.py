import pandas as pd
import joblib

model = joblib.load("../models/hdmap_rf_pipeline.joblib")

new_roads = pd.DataFrame([
    {
        "road_id":"NEW001",
        "road_type":"primary",
        "speed_limit":60,
        "lane_num":4,
        "road_length_m":3000
    },

    {
        "road_id": "NEW002",
        "road_type": "secondary",
        "speed_limit": 40,
        "lane_num": 2,
        "road_length_m": 1500
    },

    {
        "road_id": "NEW003",
        "road_type": "primary",
        "speed_limit": 200,
        "lane_num": 4,
        "road_length_m": 3000
    },

    {
        "road_id": "NEW004",
        "road_type": "secondary",
        "speed_limit": 40,
        "lane_num": 10,
        "road_length_m": 1500
    },

    {
        "road_id": "NEW005",
        "road_type": "residential",
        "speed_limit": 30,
        "lane_num": 1,
        "road_length_m": 1
    },
])

feature_columns = [
    "speed_limit",
    "lane_num",
    "road_length_m",
    "road_type"
]

X_new = new_roads[feature_columns]
predictions = model.predict(X_new)

probabilities = model.predict_proba(X_new)

new_roads["predicted_anomaly"]= predictions
new_roads["anomaly_probability"] = probabilities[:,1]

print(new_roads)

new_roads["prediction_label"] = new_roads["predicted_anomaly"].map(
    {
        0:"Normal",
        1:"Anomaly"
    })

new_roads.to_csv("../outputs/new_road_predictions.csv",index=False)
