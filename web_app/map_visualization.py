import folium

def create_road_risk_map (gdf,result_df):
    map_gdf = gdf.copy()

    if map_gdf.crs is None:
        raise ValueError("GeoDataFrame 没有CRS，无法安全绘制地图。")

    if map_gdf.crs.to_epsg() != 4326:
        map_gdf = map_gdf.to_crs("EPSG:4326")


    map_gdf["rule_anomaly"] = result_df["rule_anomaly"].values

    map_gdf["ai_predicted_anomaly"] = result_df["ai_predicted_anomaly"].values

    map_gdf["ai_anomaly_probability"] = result_df["ai_anomaly_probability"].values

    map_gdf["combined_risk"] = result_df["combined_risk"].values

    drawable_gdf = map_gdf[map_gdf.geometry.notna()].copy()

    drawable_gdf = drawable_gdf[~drawable_gdf.geometry.is_empty].copy()

    if drawable_gdf.empty:
        raise ValueError("当前数据没有可以在地图上绘制的道路geometry。")


    bounds = drawable_gdf.total_bounds

    min_lon = bounds[0]
    min_lat = bounds[1]
    max_lon = bounds[2]
    max_lat = bounds[3]

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    road_map = folium.Map(location = [center_lat,center_lon], zoom_start=11, tiles="CartoDB positron")

    normal_layer = folium.FeatureGroup(name="正常道路")
    risk_layer = folium.FeatureGroup(name="风险道路")

    for _, row in drawable_gdf.iterrows():
        geometry = row["geometry"]

        road_id = row.get("road_id","Unknown")
        road_type = row.get("road_type","Unknown")
        road_anomaly = int(row["rule_anomaly"])
        ai_anomaly = int(row["ai_predicted_anomaly"])
        probability = float(row["ai_anomaly_probability"])
        combined_risk = int(row["combined_risk"])

        if combined_risk ==1:
            road_status = "Risk"
            road_color = "red"
            target_layer = risk_layer
        else:
            road_status = "Normal"
            road_color = "green"
            target_layer = normal_layer

        coordinates = [
            [latitude,longitude]
            for longitude, latitude in geometry.coords]

        popup_text = (
            f"<b>Road ID:</b> {road_id}<br>"
            f"<b>Road Type:</b> {road_type}<br>"
            f"<b>Status:</b> {road_status}<br>"
            f"<b>Rule Anomaly:</b> {road_anomaly}<br>"
            f"<b>AI Anomaly:</b> {ai_anomaly}<br>"
            f"<b>AI Probability:</b> {probability:.2f}"
        )

        folium.PolyLine(locations=coordinates,
                        color = "#333333",
                        weight =10,
                        opacity = 0.8).add_to(target_layer)

        folium.PolyLine(
            locations=coordinates,
            color=road_color,
            weight=6,
            opacity=1.0,
            popup=folium.Popup(popup_text,max_width=350),
            tooltip=(
                f"{road_id} | "
                f"{road_status} | "
                f"AI {probability:.0%}"
                )
            ).add_to(target_layer)

        # midpoint = geometry.interpolate(0.5,normalized=True)
        # folium.CircleMarker(
        #     location=[midpoint.y,midpoint.x],
        #     radius=6,
        #     color="#222222",
        #     weight=2,
        #     fill=True,
        #     fill_color=road_color,
        #     fill_opacity=1.0,
        #     tooltip=(f"点击查看{road_id}"),
        #     popup=folium.Popup(popup_text,max_width=350)
        #     ).add_to(target_layer)

    normal_layer.add_to(road_map)
    risk_layer.add_to(road_map)
    folium.LayerControl(collapsed=False).add_to(road_map)

    road_map.fit_bounds([[min_lat,min_lon],[max_lat,max_lon]])

    return road_map