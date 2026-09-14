"""
高精地图拓扑质量检测模块

主要功能：
1. 提取道路起点和终点
2. 分析道路之间的端点邻近关系
3. 检测疑似道路断裂
4. 检测道路空间交叉关系
"""

import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from spatial_processing import(load_roads,convert_geometry,create_geodataframe)

def get_endpoints(geometry):
    """
    提取LineString的起点和终点
    """
    start_coord = geometry.coords[0]
    end_coord = geometry.coords[-1]

    start_point = Point(start_coord)
    end_point = Point(end_coord)

    return start_point,end_point

def build_endpoint_table(gdf):
    """
    将有效道路拆分成起点和终点。
    空geometry自动跳过。
    """
    endpoint_records = []

    for _, row in gdf.iterrows():

        if row.geometry is None:
            continue
        if row.geometry.is_empty:
            continue

        start_point,end_point = get_endpoints(row.geometry)

        endpoint_records.append(
            {
                "road_id":row["road_id"],
                "endpoint_type":"start",
                "geometry":start_point
            }
        )

        endpoint_records.append(
            {
                "road_id":row["road_id"],
                "endpoint_type":"end",
                "geometry":end_point
            }
        )

    endpoint_gdf = gpd.GeoDataFrame(
        endpoint_records,
        geometry="geometry",
        crs=gdf.crs
    )

    return endpoint_gdf

def check_dangling_endpoints(endpoint_gdf,tolerance=10):
    """
    检测疑似悬挂端点。

    如果某个道路端点与其他道路端点距离很近，
    但并未完全重合，则认为可能存在拓扑断裂。
    """
    errors = []
    for i, endpoint_a in endpoint_gdf.iterrows():
        for j, endpoint_b in endpoint_gdf.iterrows():
            if i >= j :
                continue
            if (endpoint_a["road_id"] == endpoint_b["road_id"]):
                continue

            distance = (endpoint_a.geometry.distance(endpoint_b.geometry))

            if 0 < distance <= tolerance:
                errors.append({
                    "road_id_a":endpoint_a["road_id"],
                    "road_id_b":endpoint_b["road_id"],
                    "endpoint_a":endpoint_a["endpoint_type"],
                    "endpoint_b":endpoint_b["endpoint_type"],
                    "distance_m":distance,
                    "error_type":"possible_topology_gap"
                })
    return pd.DataFrame(errors)

def check_road_intersections(gdf):
    """
    检测不同道路之间是否发生空间相交
    """
    intersections = []

    for i, road_a in gdf.iterrows():
        for j, road_b in gdf.iterrows():

            if i >= j:
                continue
            if road_a.geometry.intersects(road_b.geometry):

                intersection_geometry = (road_a.geometry.intersection(road_b.geometry))
                intersections.append({
                    "road_id_a":road_a["road_id"],
                    "road_id_b":road_b["road_id"],
                    "intersection_geometry":intersection_geometry,
                    "relation_type":"road_intersection"
                })
    return pd.DataFrame(intersections)

def check_road_intersections_with_sindex(gdf):
    """
    使用空间索引检测道路相交关系
    """
    intersections = []
    spatial_index = gdf.sindex
    for i, road_a in gdf.iterrows():
        candidate_indices = list(spatial_index.intersection(road_a.geometry.bounds))

        for j in candidate_indices:
            if i >= j:
                continue
            road_b =gdf.loc[j]

            if road_a.geometry.intersects(road_b.geometry):
                intersection_geometry =(road_a.geometry.intersection(road_b.geometry))

                intersections.append({
                    "road_id_a":road_a["road_id"],
                    "road_id_b":road_b["road_id"],
                    "intersection_geometry":intersection_geometry,
                    "relation_type":"road_intersection"
                })
    return pd.DataFrame(intersections)

if __name__ == "__main__":

    roads_df = load_roads("data/raw/roads_structured.csv")
    roads_df = convert_geometry(roads_df)
    roads_gdf = create_geodataframe(roads_df)
    roads_projected = roads_gdf.to_crs("EPSG:3857")
    endpoints = build_endpoint_table(roads_projected)
    topology_errors = check_dangling_endpoints(endpoints,tolerance=10)

    print("道路数量：", len(roads_projected))
    print("端点数量：", len(endpoints))
    print("疑似拓扑异常数量：", len(topology_errors))

    if topology_errors.empty:
        print("当前未发现疑似道路端点断裂。")

    else:
        print(topology_errors.head(20))

    road_intersections = check_road_intersections(roads_projected)

    print("道路相交关系数量：", len(road_intersections))

    if not road_intersections.empty:
        print(road_intersections.head(20))