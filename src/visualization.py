import geopandas as gpd
import matplotlib.pyplot as plt
from spatial_processing import (load_roads, convert_geometry, create_geodataframe)
import os
print("当前工作目录：", os.getcwd())
print("目标路径：", os.path.abspath("../raw/roads_structured.csv"))

def plot_roads(file_name="roads.csv"):

    df = load_roads(file_name)

    df = convert_geometry(df)

    gdf = create_geodataframe(df)

    gdf.plot(figsize=(10,10),linewidth=1)

    plt.title("Simulated Structured HD Map Roads")
    plt.show()

if __name__ == "__main__":

    plot_roads("./raw/roads_structured.csv")