"""
高精地图异常注入模块

功能：
1. 读取干净的结构化道路数据
2. 人为注入可控属性异常
3. 保存异常数据集
4. 为质量检测和机器学习实验提供测试数据
"""
from pathlib import Path
import pandas as pd
from shapely.geometry import LineString

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    /"raw"
    /"roads_structured.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    /"processed"
    /"roads_with_all_anomalies.csv"
)

def load_clean_roads():
    """
    读取干净的基准道路数据
    """
    return pd.read_csv(INPUT_PATH)

def inject_attribute_anomalies(df):
    """
    向道路数据中注入可控的属性异常
    为避免修改原始DataFrame，先创建副本。
    """
    anomaly_df = df.copy()

    # R0001：限速缺失
    anomaly_df.loc[
        anomaly_df["road_id"] == "R0001",
        "speed_limit"
    ] = None

    # R0002：非法道路类型
    anomaly_df.loc[
        anomaly_df["road_id"] == "R0002",
        "road_type"
    ] = "motorway"

    # R0003：限速明显超出primary规则范围
    anomaly_df.loc[
        anomaly_df["road_id"] == "R0003",
        "speed_limit"
    ] = 200

    # R0004：车道数明显超出secondary规则范围
    anomaly_df.loc[
        anomaly_df["road_id"] == "R0004",
        "lane_num"
    ] = 10

    # R0005：道路类型缺失
    anomaly_df.loc[
        anomaly_df["road_id"] == "R0005",
        "road_type"
    ] = None

    return anomaly_df

def save_anomaly_data(df):
    """
    保存异常数据集
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)


def inject_geometry_anomalies(df):
    """
    注入几何类异常
    """
    anomaly_df = df.copy()

    #1. 制造重复道路
    duplicate_road = anomaly_df[anomaly_df["road_id"]=="R0008"].copy()

    duplicate_road["road_id"] = "R0018"

    anomaly_df = pd.concat([anomaly_df,duplicate_road], ignore_index=True)

    #2.制造空geometry
    anomaly_df.loc[anomaly_df["road_id"] == "R0002","geometry"] = None

    # 3. 制造极短道路
    anomaly_df.loc[anomaly_df["road_id"] == "R0003","geometry"] ="LINESTRING " "(121.400000 31.200000, " "121.400001 31.200001)"

    return anomaly_df


def inject_topology_anomalies(df):
    """
    注入可控的道路端点拓扑断裂异常。
    """

    anomaly_df = df.copy()

    # 第一条测试道路
    road_a = {
        "road_id": "R0019",
        "road_type": "secondary",
        "speed_limit": 40,
        "lane_num": 2,
        "geometry": "LINESTRING " "(121.450000 31.250000, ""121.460000 31.250000)"
         }

    # 第二条道路原本应该接在R0019终点
    # 这里故意稍微偏移，制造拓扑gap
    road_b = {
        "road_id": "R0020",
        "road_type": "secondary",
        "speed_limit": 40,
        "lane_num": 2,
        "geometry": "LINESTRING ""(121.460050 31.250000, ""121.470000 31.250000)"
            }

    topology_test_df = pd.DataFrame([road_a, road_b])

    anomaly_df = pd.concat([anomaly_df,topology_test_df],ignore_index=True)

    return anomaly_df


if __name__ == "__main__":

    clean_df = load_clean_roads()

    # 1. 注入属性异常
    anomaly_df = inject_attribute_anomalies(clean_df)

    # 2. 注入几何异常
    anomaly_df = inject_geometry_anomalies(anomaly_df)

    # 3. 注入拓扑异常
    anomaly_df = inject_topology_anomalies(anomaly_df)

    # 4. 保存最终异常数据
    save_anomaly_data(anomaly_df)

    print ("原始道路数：",len(clean_df))
    print("异常数据道路数：",len(anomaly_df))
    print("异常数据已保存至：")
    print(OUTPUT_PATH)

