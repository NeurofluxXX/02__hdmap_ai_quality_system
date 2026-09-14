from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_roads(
            relative_path = "data/raw/roads_structured.csv"
        ):
        """
        根据项目根目录读取道路数据
        """
        file_path = PROJECT_ROOT / relative_path

        return pd.read_csv(file_path)

def convert_geometry(df):
    """
    将WKT字符串转换为Shapely几何对象。

    对缺失geometry保留为空值，
    避免wkt.loads解析NaN时报错。
    """
    def safe_wkt_loads(value):
        """
        安全解析WKT字符串。
        """
        if pd.isna(value):
            return None
        try:
            return wkt.loads(value)
        except Exception:
            return None

    df["geometry"] = df["geometry"].apply(safe_wkt_loads)
    return df

def create_geodataframe(df):
    gdf =gpd.GeoDataFrame(df,geometry="geometry",crs="EPSG:4326")
    return gdf

if __name__ == "__main__":
    df = load_roads()
    df = convert_geometry(df)
    gdf = create_geodataframe(df)

    print(gdf.head())
    print(gdf.geometry.type)
