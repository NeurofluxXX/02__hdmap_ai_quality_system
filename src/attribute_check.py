"""
高精地图属性质量检测模块

主要检测：
1. 关键字段是否缺失
2. 道路类型是否合法
3. 限速与道路类型是否匹配
4. 车道数与道路类型是否匹配
"""

import pandas as pd

from spatial_processing import load_roads

ROAD_RULES = {
    "motorway":{
        "speed_min": 80,
        "speed_max": 120,
        "lane_min": 4,
        "lane_max": 8
    },
    "primary":{
        "speed_min": 40,
        "speed_max": 80,
        "lane_min": 2,
        "lane_max": 6
    },
    "secondary":{
        "speed_min": 30,
        "speed_max": 60,
        "lane_min": 2,
        "lane_max": 4
    },
    "residential":{
        "speed_min": 10,
        "speed_max": 40,
        "lane_min": 1,
        "lane_max": 2
    }
}

def check_missing_attributes(df):
    """
    检测关键属性字段缺失
    """

    required_columns = [
        "road_id",
        "road_type",
        "speed_limit",
        "lane_num"
    ]

    missing_mask = df[required_columns].isna().any(axis=1)

    result = df[missing_mask].copy()
    result["error_type"] = "missing_attribute"

    return result

def check_invalid_road_type(df):
    """
    检测未知道路类型
    """

    valid_types = list(ROAD_RULES.keys())

    invalid_mask = ~df["road_type"].isin(valid_types)

    result = df[invalid_mask].copy()

    result["error_type"] = "invalid_road_type"

    return result

def check_rule_conflicts(df):
    """
    检测道路类型与限速、车道数之间的冲突
    """

    errors = []

    for _, row in df.iterrows():

        road_type = row["road_type"]

        if road_type not in ROAD_RULES:
            continue

        rules = ROAD_RULES[road_type]

        speed_limit = row["speed_limit"]
        lane_num = row["lane_num"]

        if pd.notna(speed_limit):

            speed_is_invalid = (
                speed_limit < rules["speed_min"]
                or
                speed_limit > rules["speed_max"]
            )

            if speed_is_invalid:

                error_record = row.to_dict()
                error_record["error_type"] = ("speed_limit_conflict")

                errors.append(error_record)

        if pd.notna(lane_num):

            lane_is_invalid = (
                lane_num < rules["lane_min"]
                or
                lane_num > rules["lane_max"]
            )

            if lane_is_invalid:
                error_record = row.to_dict()
                error_record["error_type"] = "lane_num_conflict"

                errors.append(error_record)

    return pd.DataFrame(errors)

def run_attribute_checks(df):
    """
    执行全部属性质量检查
    """

    missing_errors = check_missing_attributes(df)
    type_errors = check_invalid_road_type(df)
    rule_errors = check_rule_conflicts(df)

    all_errors = pd.concat([missing_errors, type_errors, rule_errors], ignore_index=True)

    return all_errors

if __name__ == "__main__":
    roads_df = load_roads("data/processed/roads_with_anomalies.csv")

    attribute_errors = run_attribute_checks(roads_df)

    print("道路总数：", len(roads_df))
    print("属性异常数", len(attribute_errors))

    if attribute_errors.empty:
        print("当前未发现属性质量问题。")
    else:
        print(
            attribute_errors[
                [
                    "road_id",
                    "road_type",
                    "speed_limit",
                    "lane_num",
                    "error_type"
                ]
            ]
        )


