import torch

# training and inference device
GPU_NUMBER = torch.cuda.device_count()

# features
TIME_COL = ["datetime_id"]
ROAD_COL = ["section_id"]
SEARCH_COL = ["search_1h", "search_unspec_1d"]
TRAFFIC_COL = ["allCars"]
FEATURE_COL = TIME_COL + ROAD_COL + SEARCH_COL + TRAFFIC_COL
TARGET_COL = "allCars"

DT_TABLE_SIZE = 288
SEC_TABLE_SIZE = 63

# column num of static features (time id & road id)
TIME_COL_INDEX = 0
ROAD_COL_INDEX = 1
STATIC_COL = [TIME_COL_INDEX, ROAD_COL_INDEX]
SEARCH_COL_INDEX = [-3, -2]
TRAFFIC_COL_INDEX = [-1]