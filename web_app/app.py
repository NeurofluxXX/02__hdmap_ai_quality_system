#streamlit run app.py
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from attribute_check import run_attribute_checks
from spatial_processing import convert_geometry,create_geodataframe
from geometry_check import run_geometry_checks
from topology_check import build_endpoint_table,check_dangling_endpoints
from quality_report import build_quality_summary
from feature_engineering import add_length_feature
from feature_engineering import add_length_feature
from map_visualization import create_road_risk_map
from streamlit_folium import st_folium

st.set_page_config(page_title="HDMap-AI-QA",page_icon="🗺️",layout="wide")

st.title("HDMap-AI-QA")
st.subheader("High Definition Map Quality Assurance System")
st.write("高精地图规则质量检测与AI异常识别系统")

MODEL_PATH = (MODELS_DIR / "hdmap_rf_pipeline.joblib")

model = joblib.load(MODEL_PATH)

def run_rule_quality_check(df):
    attribute_errors = run_attribute_checks(df)

    spatial_df = convert_geometry(df.copy())
    gdf = create_geodataframe(spatial_df)

    geometry_errors = run_geometry_checks(gdf)

    projected_gdf = gdf.to_crs("EPSG:3857")

    endpoint_gdf = build_endpoint_table(projected_gdf)

    topology_errors = check_dangling_endpoints(endpoint_gdf,tolerance=10)

    total_roads = len(df)
    summary = build_quality_summary(total_roads,
                                    attribute_errors,
                                    geometry_errors,
                                    topology_errors)

    return {
        "attribute_errors": attribute_errors,
        "geometry_errors": geometry_errors,
        "topology_errors": topology_errors,
        "summary": summary}

st.divider()
st.subheader("AI 检测参数")

threshold = st.slider(
    "AI 异常判定阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

uploaded_file = st.file_uploader("上传道路csv文件",type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("道路数据读取成功")

    st.subheader("道路数据预览")
    st.dataframe(df.head(20),use_container_width=True)

    feature_columns = [
        "road_id",
        "road_type",
        "speed_limit",
        "lane_num",
        "geometry"
    ]

    missing_columns = [column
                       for column in feature_columns
                       if column not in df.columns]

    if missing_columns:
        st.error(f"缺少必要字段：{missing_columns}")

    else:
        st.divider()
        st.subheader("规则质量检测结果")

        rule_results = run_rule_quality_check(df)

        summary = rule_results["summary"]

        spatial_df = convert_geometry(df.copy())

        gdf = create_geodataframe(spatial_df)

        feature_gdf = add_length_feature(gdf)

        ai_feature_columns = [
            "speed_limit",
            "lane_num",
            "road_length_m",
            "road_type"
        ]

        X_ai = feature_gdf[ai_feature_columns].copy()

        X_ai["speed_limit"] = X_ai["speed_limit"].fillna(X_ai["speed_limit"].median())
        X_ai["lane_num"] = X_ai["lane_num"].fillna(X_ai["lane_num"].median())
        X_ai["road_length_m"] = X_ai["road_length_m"].fillna(0)
        X_ai["road_type"] = X_ai["road_type"].fillna("unknown")

        ai_probabilities = model.predict_proba(X_ai)[:,1]

        ai_predictions = (ai_probabilities >= threshold).astype(int)

        result_df = df.copy()

        result_df["road_length_m"] = feature_gdf["road_length_m"]
        result_df["ai_anomaly_probability"] = ai_probabilities
        result_df["ai_predicted_anomaly"] = ai_predictions
        result_df["ai_prediction_label"] = result_df["ai_predicted_anomaly"].map({
            0:"Normal",
            1:"Anomaly"
        })

        rule_abnormal_ids = set()
        # ---------------- 属性异常 ----------------
        attribute_errors = rule_results["attribute_errors"]

        if not attribute_errors.empty:
            rule_abnormal_ids.update(attribute_errors["road_id"].dropna().tolist())
        # ---------------- 几何异常 ----------------
        geometry_errors = rule_results["geometry_errors"]

        if not geometry_errors.empty:
            rule_abnormal_ids.update(geometry_errors["road_id"].dropna().tolist())
        # ---------------- 拓扑异常 ----------------
        topology_errors = rule_results["topology_errors"]
        if not topology_errors.empty:
            for column in ["road_id_a","road_id_b"]:
                if column in topology_errors.columns:
                    rule_abnormal_ids.update(topology_errors[column].dropna().tolist())

        # 18. 给每条道路增加 rule_anomaly
        result_df["rule_anomaly"] = result_df["road_id"].isin(rule_abnormal_ids).astype(int)

        # 19. 综合规则检测和 AI 检测
        result_df["combined_risk"] = ((result_df["rule_anomaly"] == 1) | (result_df["ai_predicted_anomaly"] ==1)).astype(int)

        # 20. 页面顶部核心统计指标
        st.divider()
        st.subheader("质量检测概览")

        # 创建5个横向指标卡片。
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("道路总数",summary["total_roads"])

        with col2:
            st.metric("属性异常事件：",summary["attribute_error_count"])

        with col3:
            st.metric("几何异常事件",summary["geometry_error_count"])

        with col4:
            st.metric("拓扑异常事件",summary["topology_error_count"])

        with col5:
            st.metric("AI异常道路",int(ai_predictions.sum()))

        st.metric("规则质量评分",f"{summary['quality_score']:.1f}%")

        # 道路空间风险地图
        st.divider()
        st.subheader("道路空间风险地图")

        road_map = create_road_risk_map(gdf,result_df)

        st_folium(road_map,width=None,height=550,returned_objects=[])

        st.caption("红色表示综合风险道路，绿色表示正常道路；"
                    "所有带深色边框的线均为本系统正在检测的模拟道路。"
                    "可点击道路中点圆圈或道路本身查看规则检测结果与AI异常概率。")


        # 21. 创建异常详情 Tabs
        st.divider()
        st.subheader("异常检测详情")

        # tabs 可以让多个结果显示在不同标签页中。
        tab1,tab2,tab3,tab4 = st.tabs([
            "属性异常",
            "几何异常",
            "拓扑异常",
            "AI异常"
        ])

        # ---------------- 属性异常 Tab ----------------
        with tab1:
            if attribute_errors.empty:
                st.success("未发现属性异常")
            else:
                st.dataframe(attribute_errors,use_container_width=True)

        # ---------------- 几何异常 Tab ----------------
        with tab2:
            if geometry_errors.empty:
                st.success("未发现几何异常")
            else:
                st.dataframe(geometry_errors,use_container_width=True)

        # ---------------- 拓扑异常 Tab ----------------
        with tab3:
            if topology_errors.empty:
                st.success("未发现拓扑异常")
            else:
                st.dataframe(topology_errors,use_container_width=True)

        # ---------------- AI异常 Tab ----------------
        with tab4:
            ai_anomaly_df = result_df[result_df["ai_predicted_anomaly"] ==1].copy()

            ai_anomaly_df = ai_anomaly_df.sort_values("ai_anomaly_probability", ascending=False)

            if ai_anomaly_df.empty:
                st.success("AI模型未发现异常道路")

            else:
                st.dataframe(ai_anomaly_df,use_container_width=True)

        # 22. 综合风险道路
        st.divider()
        st.subheader("综合风险道路")

        combined_risk_df = result_df[result_df["combined_risk"]==1].copy()
        combined_risk_df = combined_risk_df.sort_values("ai_anomaly_probability", ascending=False)

        if combined_risk_df.empty:
            st.success("当前未发现综合风险道路")

        else:
            st.dataframe(combined_risk_df,use_container_width=True)

        # 23. 显示完整检测结果
        st.divider()
        st.subheader("完整检测结果")

        st.dataframe(result_df,use_container_width=True)

        # 24. 下载检测结果
        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(label="📥 下载检测结果 CSV",
                           data=csv_data,
                           file_name="hdmap_quality_result.csv",mime="text/csv")

        # 25. 实验说明
        st.divider()
        st.caption(
            "说明：规则质量评分为本项目设计的实验性指标；"
            "AI异常概率来自模拟道路数据训练的模型，"
            "主要用于验证完整GeoAI工程流程，"
            "不代表真实生产高精地图数据上的实际风险概率。"
        )
