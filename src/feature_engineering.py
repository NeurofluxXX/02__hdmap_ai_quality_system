"""
HD Map机器学习特征工程模块

功能：
1. 读取道路数据
2. 计算道路空间特征
3. 编码道路属性
4. 构建机器学习训练数据
"""
import pandas as pd
from spatial_processing import load_roads,convert_geometry,create_geodataframe
from attribute_check import run_attribute_checks
from geometry_check import run_geometry_checks
from topology_check import build_endpoint_table,check_dangling_endpoints

def add_length_feature(gdf):
    """
    计算道路长度特征，单位：米
    """
    projected_gdf = gdf.to_crs("EPSG:3857")
    gdf = gdf.copy()
    gdf["road_length_m"] = projected_gdf.geometry.length
    return gdf

def encode_road_type(df):
    """
    对road_type进行One-Hot Encoding
    """
    encoded_df = pd.get_dummies(df,columns=["road_type"],dtype=int)
    return encoded_df

def handle_missing_values(df):
    """
    处理机器学习特征中的缺失值
    """
    df = df.copy()
    df["speed_limit"] = df["speed_limit"].fillna(df["speed_limit"].median())
    df["lane_num"] = df["lane_num"].fillna(df["lane_num"].median())
    df["road_length_m"] = df["road_length_m"].fillna(0)
    return df

def build_anomaly_labels(df,attribute_errors,geometry_errors,topology_errors):
    """
    根据规则检测结果生成机器学习标签。
    """
    abnormal_ids = set()

    if not attribute_errors.empty:
        abnormal_ids.update(attribute_errors["road_id"].dropna().tolist())

    if not geometry_errors.empty:
        abnormal_ids.update(geometry_errors["road_id"].dropna().tolist())

    if not topology_errors.empty:
        if "road_id_a" in topology_errors.columns:
            abnormal_ids.update(topology_errors["road_id_a"].dropna().tolist())
        if "road_id_b" in topology_errors.columns:
            abnormal_ids.update(topology_errors["road_id_b"].dropna().tolist())

    df = df.copy()

    df["is_anomaly"] = df["road_id"].isin(abnormal_ids).astype(int)
    return df

def save_ml_dataset(df):
    """
    保存机器学习数据集
    """
    output_columns = []
    for column in df.columns:
        if column != "geometry":
            output_columns.append(column)

    df[output_columns].to_csv("../data/processed/ml_dataset.csv",index=False)

def build_ml_dataset(input_path="data/processed/roads_with_all_anomalies.csv"):
    """
    构建完整机器学习数据集
    """
    # 1. 读取原始道路数据
    df = load_roads(input_path)

    # 2. 先运行属性质量检测
    attribute_errors = run_attribute_checks(df)

    # 3. 转换geometry
    spatial_df = convert_geometry(df.copy())

    gdf = create_geodataframe(spatial_df)

    # 4. 几何质量检测
    geometry_errors = run_geometry_checks(gdf)

    # 5. 拓扑质量检测
    projected_gdf = gdf.to_crs("EPSG:3857")

    endpoint_gdf = build_endpoint_table(projected_gdf)

    topology_errors = check_dangling_endpoints(endpoint_gdf,tolerance=10)

    # 6. 添加长度特征
    feature_gdf = add_length_feature(gdf)

    # 7. 根据质量检测结果生成label
    feature_gdf = build_anomaly_labels(feature_gdf,attribute_errors,geometry_errors,topology_errors)

    # 8. 处理缺失值
    feature_gdf = handle_missing_values(feature_gdf)

    # 9. One-Hot编码
    ml_df = encode_road_type(feature_gdf)

    # 10. 保存
    save_ml_dataset(ml_df)

    return ml_df

if __name__ == "__main__":
    ml_df = build_ml_dataset()
    print("\n========== ML Dataset ==========")
    print(ml_df.head())
    print("\n数据集形状：",ml_df.shape)
    print("\n标签分布：")
    print(ml_df["is_anomaly"].value_counts())
