"""
高精地图几何质量检测模块

主要检测：
1. geometry 是否缺失
2. geometry 是否为空
3. geometry 是否有效
4. 是否存在重复道路几何
"""

import pandas as pd
from spatial_processing import (load_roads, convert_geometry, create_geodataframe)

def load_spatial_roads(file_name="roads_structured.csv"):
    """
    读取道路文件并转换为 GeoDataFrame
    """

    df = load_roads(file_name)
    df = convert_geometry(df)
    gdf = create_geodataframe(df)

    return gdf

def check_missing_geometry(gdf):
    """
    检测 geometry 为缺失值的道路
    """
    result = gdf[gdf["geometry"].isna()].copy()

    result["error_type"] = "missing_geometry"

    return result

def check_empty_geometry(gdf):
    """
    检测空的几何对象
    """
    result = gdf[gdf.geometry.is_empty].copy()
    result["error_type"] = "empty_geometry"
    return result

def check_invalid_geometry(gdf):
    """
    检测存在几何对象、但几何本身无效的记录。

    缺失geometry已经由missing_geometry负责，
    因此这里排除空值。
    """
    not_missing = gdf.geometry.notna()
    invalid = ~gdf.geometry.is_valid
    result = gdf[not_missing & invalid].copy()

    result["error_type"] = "invalid_geometry"

    return result

def check_duplicate_geometry(gdf):
        """
        检测完全重复的有效道路几何。

        使用WKB进行精确几何比较。
        缺失geometry不参与检测。
        """

        valid_gdf = gdf[gdf.geometry.notna()].copy()

        geometry_wkb = valid_gdf.geometry.to_wkb(hex=True)

        duplicated_mask = geometry_wkb.duplicated(keep=False)


        result = valid_gdf[duplicated_mask].copy()

        result["error_type"] = "duplicate_geometry"

        return result

def run_geometry_checks(gdf):
    """
    执行全部几何质量检查
    """
    missing_errors = check_missing_geometry(gdf)
    empty_errors = check_empty_geometry(gdf)
    invalid_errors = check_invalid_geometry(gdf)
    duplicate_errors = check_duplicate_geometry(gdf)
    short_errors = check_short_geometry(gdf)

    all_errors = pd.concat([
        missing_errors,
        empty_errors,
        invalid_errors,
        duplicate_errors,
        short_errors
    ],
    ignore_index=True
    )

    return all_errors

def debug_duplicate_geometry(gdf):
    """
    调试重复geometry，并为相同geometry建立分组编号。
    """

    valid_gdf = (gdf[gdf.geometry.notna()].copy())

    valid_gdf["geometry_wkb"] = (valid_gdf.geometry.to_wkb(hex=True))

    duplicate_mask = (valid_gdf["geometry_wkb"].duplicated(keep=False))

    duplicate_debug = (valid_gdf[duplicate_mask].copy())

    duplicate_debug["duplicate_group"] = duplicate_debug.groupby("geometry_wkb").ngroup() + 1

    return duplicate_debug[["road_id","duplicate_group","geometry"]].sort_values(["duplicate_group","road_id"])

def check_short_geometry(gdf,min_length_m=5):
    """
    检测长度异常过短的道路。
    """
    valid_gdf = gdf[gdf.geometry.notna()].copy()
    projected_gdf = valid_gdf.to_crs("EPSG:3857")
    short_mask = projected_gdf.geometry.length < min_length_m
    result = valid_gdf.loc[short_mask.values].copy()
    result["error_type"] = "short_geometry"

    return result

if __name__ == "__main__":

    roads_gdf = load_spatial_roads("D:\\Project\\geo-ai-career-portfolio\\03__hdmap_ai_quality_system\\data\\raw\\roads_structured.csv")

    geometry_errors = run_geometry_checks(roads_gdf)

    print("道路总数：", len(roads_gdf))
    print("几何异常数", len(geometry_errors))

    if geometry_errors.empty:
        print("当前未发现几何质量问题。")
    else:
        print(geometry_errors[["road_id", "error_type", "geometry"]])

    duplicate_debug = (
        debug_duplicate_geometry(
            roads_gdf
        )
    )

    print("\n========== 重复几何诊断 ==========")

    print(duplicate_debug.to_string(index=False))
