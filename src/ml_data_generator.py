"""
机器学习道路样本生成模块
"""

import random
import pandas as pd


ROAD_PROFILES = {

    "motorway": {
        "speed_choices": [80, 100, 120],
        "lane_choices": [4, 6, 8],
        "length_range": (1000, 10000)
    },

    "primary": {
        "speed_choices": [40, 60, 80],
        "lane_choices": [2, 4, 6],
        "length_range": (500, 6000)
    },

    "secondary": {
        "speed_choices": [30, 40, 50, 60],
        "lane_choices": [2, 3, 4],
        "length_range": (200, 4000)
    },

    "residential": {
        "speed_choices": [20, 30, 40],
        "lane_choices": [1, 2],
        "length_range": (50, 2000)
    }
}

def generate_normal_sample(index):
    road_type = random.choice(list(ROAD_PROFILES.keys()))
    profile = ROAD_PROFILES[road_type]
    speed_limit = random.choice(profile["speed_choices"])
    lane_num = random.choice(profile["lane_choices"])
    min_length,max_length = (profile["length_range"])
    road_length_m = random.uniform(min_length,max_length)

    return {"road_id":f"MLR{index:05d}",
            "road_type":road_type,
            "speed_limit":speed_limit,
            "lane_num":lane_num,
            "road_length_m":round(road_length_m,2),
            "is_anomaly":0}

def inject_ml_anomaly(sample):
    sample = sample.copy()
    anomaly_type = random.choice([
        "speed",
        "lane",
        "short",
        "attribute_combo"
    ])

    if anomaly_type == "speed":
        sample["speed_limit"] = random.choice([150,180,200])

    elif anomaly_type =="lane":
        sample["lane_num"] = random.choice([9,10,12])

    elif anomaly_type =="short":
        sample["road_length_m"] = random.uniform(0.1,5)

    elif anomaly_type == "attribute_combo":
        sample["speed_limit"] =120
        sample["lane_num"] = 1

    sample["is_anomaly"] = 1

    sample["anomaly_type"] = anomaly_type

    return sample

def generate_ml_dataset(num_samples=1000,anomaly_ratio=0.2,random_seed=42):
    random.seed(random_seed)
    samples = []

    for i in range (num_samples):
        sample = generate_normal_sample(i+1)

        if random.random()<anomaly_ratio:
            sample = inject_ml_anomaly(sample)

        else:
            sample["anomaly_type"] = "normal"

        samples.append(sample)

    return pd.DataFrame(samples)

if "__main__" == __name__:
    df = generate_ml_dataset(num_samples=1000,anomaly_ratio=0.2,random_seed=42)

    print(df.head())

    print("\n数据规模：",df.shape)

    print("\n标签分布：")
    print(df["is_anomaly"].value_counts())

    print("\n异常类型分布：")
    print(df["anomaly_type"].value_counts())

    df.to_csv("../data/processed/""ml_training_dataset.csv",index=False)

