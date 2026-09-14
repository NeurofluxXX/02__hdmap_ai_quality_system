"""
HD Map质量报告模块
"""

def calculate_quality_score(total_roads,abnormal_roads):
    """
    根据异常道路比例计算基础质量评分
    """
    if total_roads ==0:
        return 0.0
    abnormal_ratio = abnormal_roads/total_roads

    score = (1-abnormal_ratio)*100

    score = max(0,min(100,score))

    return round(score,2)

def build_quality_summary(total_roads, attribute_errors, geometry_errors, topology_errors):
    """
    汇总质量检测结果
    """
    abnormal_road_ids = set()

    for error_df in [attribute_errors,geometry_errors]:
        if not error_df.empty and "road_id" in error_df.columns:
            abnormal_road_ids.update(error_df["road_id"].dropna().tolist())

    if not topology_errors.empty:
        if "road_id_a" in topology_errors.columns :
            abnormal_road_ids.update(topology_errors["road_id_a"].dropna().tolist())

        if "road_id_b" in topology_errors.columns :
            abnormal_road_ids.update(topology_errors["road_id_b"].dropna().tolist())

    abnormal_roads = len(abnormal_road_ids)

    quality_score = calculate_quality_score(total_roads,abnormal_roads)

    summary = {
        "total_roads":total_roads,
        "abnormal_roads":abnormal_roads,
        "attribute_error_count":len(attribute_errors),
        "geometry_error_count":len(geometry_errors),
        "topology_error_count":len(topology_errors),
        "quality_score":quality_score
    }

    return summary