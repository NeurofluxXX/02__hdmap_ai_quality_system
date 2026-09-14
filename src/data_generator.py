"""
HD Map 数据生成模块

功能：
1. 模拟生成城市道路中心线数据
2. 生成车道数据
3. 生成交通标志数据

模拟区域：
上海城市道路网络

"""
import random
import pandas as pd
from shapely.geometry import LineString

def generate_road(road_id):
    """
    生成单条模拟道路
    """

    #随机生成道路起点
    start_lon = random.uniform(121.30,121.60)
    start_lat = random.uniform(31.10,31.40)

    #随机生成道路终点
    end_lon = start_lon + random.uniform(0.01,0.05)
    end_lat = start_lat + random.uniform(0.01,0.05)

    geometry = LineString(          #LineString设置道路
        [
            (start_lon,start_lat),
            (end_lon,end_lat)
        ]
    )

    road_type = random.choice([
        "motorway",
        "primary",
        "secondary",
        "residential"
    ])

    speed_limit = random.choice([30,40,60,80,100])

    lane_num = random.choice([2,3,4,6]) #车道数


    return{ "road_id":road_id,
            "road_type":road_type,
            "speed_limit":speed_limit,
            "lane_num":lane_num,
            "geometry":geometry
    }

def create_road_record(road_id, coords, road_type, speed_limit, lane_num):
        """
        根据坐标列表创建道路记录
        """
        geometry = LineString(coords)

        return{
            "road_id":road_id,
            "road_type":road_type,
            "speed_limit":speed_limit,
            "lane_num":lane_num,
            "geometry":geometry
        }

def generate_roads(num_roads):
    """
    批量生成道路
    """
    roads = []
    for i in range(num_roads):
        road_id = f"R{i+1:04d}"
        road = generate_road(road_id)
        roads.append(road)
    return roads

def roads_to_dataframe(roads):    #将道路列表转换为DataFrame
    df = pd.DataFrame(roads)
    return df

def save_road_csv(df, file_name="roads.csv"):
    df.to_csv(f"../data/raw/{file_name}",index=False)

def generate_structured_roads():
    """
    生成类城市道路网络
    包括：
    1. 南北向道路
    2. 东西向道路
    3. 少量斜向连接道路
    """
    roads = []
    road_num = 1

    # --------------------------
    # 1. 南北向道路（纵向主路）
    # --------------------------
    vertical_lons = [121.34,121.38, 121.42, 121.46, 121.50, 121.54, 121.58]

    for i, lon in enumerate(vertical_lons):
        road_id = f"R{road_num:04d}"
        road_num += 1

        coords = [
            (lon + random.uniform(-0.002, 0.002), 31.11),
            (lon + random.uniform(-0.002, 0.002), 31.20),
            (lon + random.uniform(-0.002, 0.002), 31.30),
            (lon + random.uniform(-0.002, 0.002), 31.40),
        ]

        if i % 2 == 0:
            road_type = "primary"
            speed_limit = 60
            lane_num = 4
        else:
            road_type = "secondary"
            speed_limit = 40
            lane_num = 2

        road = create_road_record(
            road_id, coords, road_type, speed_limit, lane_num
        )
        roads.append(road)


    # --------------------------
    # 2. 东西向道路（横向主路）
    # --------------------------
    horizontal_lats = [31.14, 31.18, 31.22, 31.26, 31.30, 31.34, 31.38]

    for i, lat in enumerate(horizontal_lats):
        road_id = f"R{road_num:04d}"
        road_num += 1

        coords = [
            (121.31, lat + random.uniform(-0.002, 0.002)),
            (121.40, lat + random.uniform(-0.002, 0.002)),
            (121.50, lat + random.uniform(-0.002, 0.002)),
            (121.60, lat + random.uniform(-0.002, 0.002)),
        ]

        if i % 2 == 0:
            road_type = "primary"
            speed_limit = 60
            lane_num = 4
        else:
            road_type = "secondary"
            speed_limit = 40
            lane_num = 2

        road = create_road_record(
            road_id, coords, road_type, speed_limit, lane_num
        )
        roads.append(road)

        # --------------------------
        # 3. 斜向连接道路
        # --------------------------
    diagonal_roads = [ [(121.33, 31.15), (121.40, 31.22), (121.48, 31.29), (121.57, 31.36)],
         [(121.35, 31.38), (121.43, 31.32), (121.52, 31.25), (121.60, 31.18)],
         [(121.37, 31.12), (121.45, 31.18), (121.53, 31.24), (121.59, 31.30)] ]

    for coords in diagonal_roads:
         road_id = f"R{road_num:04d}"
         road_num += 1
         road = create_road_record(road_id,coords,"secondary",50,3)
         roads.append(road)

    return roads

if __name__ == "__main__":
    roads = generate_structured_roads()
    df = roads_to_dataframe(roads)
    save_road_csv(df,"roads_structured.csv")
    print(df.head())
    print(f"共生成{len(df)}条结构化道路")


