# problem definition
TIME_STEP = 168
PREDICTION_HORIZON = 24
PREDICTION_INTERVAL = 24

SPACE_WINDOW = (-2, 2)

# data details
BATCH_SIZE = 256

# network parameters
BIDIRECTIONAL_LSTM = True

########################
# Traffic Volume Encoder
########################
# Convolution
TRAFFIC_VOLUME_CONV_DIM = 128
TRAFFIC_VOLUME_CONV_TEMPORAL_KERNEL = 7
TRAFFIC_VOLUME_CONV_SPATIAL_KERNEL = 5
TRAFFIC_VOLUME_CONV_KERNEL = (
    TRAFFIC_VOLUME_CONV_TEMPORAL_KERNEL,
    TRAFFIC_VOLUME_CONV_SPATIAL_KERNEL,
)
TRAFFIC_VOLUME_CONV_PADDING = (TRAFFIC_VOLUME_CONV_TEMPORAL_KERNEL // 2, 0)
TRAFFIC_VOLUME_CONV_PADDING_MODE = "replicate"
# LSTM
TRAFFIC_VOLUME_LSTM_DIM = 256
TRAFFIC_VOLUME_LSTM_LAYERS = 3
TRAFFIC_VOLUME_LSTM_DROPOUT = 0

TRAFFIC_VOLUME_HIDDEN = TRAFFIC_VOLUME_LSTM_DIM * (1 + int(BIDIRECTIONAL_LSTM))

######################
# Search Count Encoder
######################
# Specified Search Count Encoder
# Convolution
SPECIFIED_SEARCH_COUNT_CONV_DIM = 64
SPECIFIED_SEARCH_COUNT_CONV_TEMPORAL_KERNEL = 7
SPECIFIED_SEARCH_COUNT_CONV_SPATIAL_KERNEL = 5
SPECIFIED_SEARCH_COUNT_CONV_KERNEL = (
    SPECIFIED_SEARCH_COUNT_CONV_TEMPORAL_KERNEL,
    SPECIFIED_SEARCH_COUNT_CONV_SPATIAL_KERNEL,
)
SPECIFIED_SEARCH_COUNT_CONV_PADDING = (
    SPECIFIED_SEARCH_COUNT_CONV_TEMPORAL_KERNEL // 2,
    0,
)
SPECIFIED_SEARCH_COUNT_CONV_PADDING_MODE = "replicate"
# LSTM
SPECIFIED_SEARCH_COUNT_LSTM_DIM = 256
SPECIFIED_SEARCH_COUNT_LSTM_LAYERS = 3
SPECIFIED_SEARCH_COUNT_LSTM_DROPOUT = 0

SPECIFIED_SEARCH_COUNT_HIDDEN = SPECIFIED_SEARCH_COUNT_LSTM_DIM * (
    1 + int(BIDIRECTIONAL_LSTM)
)

# Unspecified Search Count Encoder
# Convolution
UNSPECIFIED_SEARCH_COUNT_CONV_DIM = 64
UNSPECIFIED_SEARCH_COUNT_CONV_TEMPORAL_KERNEL = 1
UNSPECIFIED_SEARCH_COUNT_CONV_SPATIAL_KERNEL = 5
UNSPECIFIED_SEARCH_COUNT_CONV_KERNEL = (
    UNSPECIFIED_SEARCH_COUNT_CONV_TEMPORAL_KERNEL,
    UNSPECIFIED_SEARCH_COUNT_CONV_SPATIAL_KERNEL,
)
UNSPECIFIED_SEARCH_COUNT_CONV_PADDING = (
    UNSPECIFIED_SEARCH_COUNT_CONV_TEMPORAL_KERNEL // 2,
    0,
)
UNSPECIFIED_SEARCH_COUNT_CONV_PADDING_MODE = "replicate"
# LSTM
UNSPECIFIED_SEARCH_COUNT_LSTM_DIM = 64
UNSPECIFIED_SEARCH_COUNT_LSTM_LAYERS = 1
UNSPECIFIED_SEARCH_COUNT_LSTM_DROPOUT = 0

UNSPECIFIED_SEARCH_COUNT_HIDDEN = UNSPECIFIED_SEARCH_COUNT_LSTM_DIM * (
    1 + int(BIDIRECTIONAL_LSTM)
)

SEARCH_COUNT_HIDDEN = SPECIFIED_SEARCH_COUNT_HIDDEN + UNSPECIFIED_SEARCH_COUNT_HIDDEN


###################
# Context Embedding
###################
ROAD_N_EMBEDDING = 8
DATETIME_N_EMBEDDING = 64
CONTEXT_N_EMBEDDING = ROAD_N_EMBEDDING + DATETIME_N_EMBEDDING
CONTEXT_HIDDEN = CONTEXT_N_EMBEDDING

CONTEXT_EMBEDDING_DROPOUT = 0


######################
# Traffic Decode Layer
######################
TRAFFIC_VOLUME_DECODE_START_VALUE = -1.0

TRAFFIC_VOLUME_DECODE_LSTM_DIM = TRAFFIC_VOLUME_LSTM_DIM * (1 + int(BIDIRECTIONAL_LSTM))
TRAFFIC_VOLUME_DECODE_LSTM_LAYERS = TRAFFIC_VOLUME_LSTM_LAYERS
TRAFFIC_VOLUME_DECODE_LSTM_DROPOUT = 0

TRAFFIC_VOLUME_DECODE_HIDDEN = TRAFFIC_VOLUME_DECODE_LSTM_DIM

######################
# Feature Concat Layer
######################
FEATURE_CONCAT_HIDDEN = (
    TRAFFIC_VOLUME_DECODE_HIDDEN + SEARCH_COUNT_HIDDEN + CONTEXT_HIDDEN
)

##############
# Affine Layer
##############
FC_DIM_LIST = [FEATURE_CONCAT_HIDDEN, 64, 16, 1]
FC_LAYERS = len(FC_DIM_LIST) - 1

# baseline parameters
ARIMA_ORDER = (1, 1, 0)
SVR_KERNEL = "rbf"
# others
RANDOM_SEED = 42
