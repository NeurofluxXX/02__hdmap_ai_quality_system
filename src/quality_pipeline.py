"""
高精地图统一质量检测Pipeline
"""
import pandas as pd

from spatial_processing import load_roads, convert_geometry,create_geodataframe
from attribute_check import run_attribute_checks
from topology_check import build_endpoint_table,check_dangling_endpoints
from geometry_check import run_geometry_checks,debug_duplicate_geometry
from quality_report import build_quality_summary

def run_quality_pipeline(input_path):
    """
    执行完整道路质量检测流程
    """

    print("========== HD Map Quality Check ==========")
    # 1. 读取数据
    df = load_roads(input_path)

    print("道路总数：",len(df))

    # 2. 属性检查
    attribute_errors = (run_attribute_checks(df))

    # 3. 空间数据转换
    spatial_df = convert_geometry(df.copy())

    gdf = create_geodataframe(spatial_df)

    # 4. 几何检查
    geometry_errors = run_geometry_checks(gdf)

    duplicate_debug = debug_duplicate_geometry(gdf)

    print("\n========== 重复geometry分组 ==========")

    if duplicate_debug.empty:print("不存在重复geometry。")
    else:
        print(duplicate_debug.to_string(index=False))

    # 5. 拓扑检查
    projected_gdf = gdf.to_crs("EPSG:3857")

    endpoint_gdf = (build_endpoint_table(projected_gdf))

    topology_errors = (check_dangling_endpoints(endpoint_gdf,tolerance=10))

    print("属性异常",len(attribute_errors))

    print("几何异常",len(geometry_errors))

    print("拓扑异常",len(topology_errors))

    print("\n========== 属性异常类型 ==========")
    if not attribute_errors.empty:
        print(attribute_errors["error_type"].value_counts())

    print("\n========== 几何异常类型 ==========")
    if not geometry_errors.empty:
        print(geometry_errors["error_type"].value_counts())

    print("\n========== 拓扑异常类型 ==========")
    if not topology_errors.empty:
        print(topology_errors["error_type"].value_counts())

    print("\n========== 几何异常详情 ==========")
    if not geometry_errors.empty:
        print(geometry_errors[
                  ["road_id","error_type"]
              ].to_string(index=False))

    print("\n========== Quality Summary ==========")
    summary = build_quality_summary(total_roads=len(df),
                                    attribute_errors=attribute_errors,
                                    geometry_errors=geometry_errors,
                                    topology_errors=topology_errors)

    for key, value in summary.items():
        print(f"{key}:{value}")

    print(
        "\n唯一geometry数量：",
        gdf.geometry.to_wkb(
            hex=True
        ).nunique(
            dropna=True
        )
    )



    return {
        "summary":summary,
        "attribute_errors":attribute_errors,
        "geometry_errors":geometry_errors,
        "topology_errors":topology_errors
    }



if __name__ == "__main__":
    results = run_quality_pipeline("data/processed/roads_with_all_anomalies.csv")