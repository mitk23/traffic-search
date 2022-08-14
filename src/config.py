# data
DATASET_DIR = "./datasets"
TABLES_DIR = f"{DATASET_DIR}/tables"
MINIDATA_DIR = f"{DATASET_DIR}/mini"

# log
LOG_DIR = "./logs"

# model
MODEL_DIR = "./models"

# features
TIME_COL = ["datetime_id"]
ROAD_COL = ["section_id"]
SEARCH_COL = ["search_15min", "search_unspec_1d"]
TRAFFIC_COL = ["allCars"]
FEATURE_COL = TIME_COL + ROAD_COL + SEARCH_COL + TRAFFIC_COL
TARGET_COL = "allCars"

# parameters
RANDOM_SEED = 42
BATCH_SIZE = 256
TIME_STEP = 96
PREDICTION_HORIZON = 1

# column num of static features (time id & road id)
STATIC_COL = [0, 1]
