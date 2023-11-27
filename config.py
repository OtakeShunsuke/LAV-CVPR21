from easydict import EasyDict as edict

CONFIG = edict()

CONFIG.DESCRIPTION = "Exp 1"

CONFIG.ROOT = "/root/src/checkpoints/Penn_Action"
CONFIG.CKPT_PATH = "/root/src/checkpoints/Penn_Action"
CONFIG.DATA_PATH = "/root/src/dataset/Penn_Action/all_divided_dataset"
CONFIG.GPUS = 1
CONFIG.SEED = 0

CONFIG.TRAIN = edict()

CONFIG.TRAIN.NUM_FRAMES = 20
CONFIG.TRAIN.EPOCHS = 100
CONFIG.TRAIN.SAVE_INTERVAL_ITERS = 1000
CONFIG.TRAIN.LR = 1e-4
CONFIG.TRAIN.WEIGHT_DECAY = 1e-5
CONFIG.TRAIN.BATCH_SIZE = 1

CONFIG.TRAIN.FREEZE_BASE = False
CONFIG.TRAIN.FREEZE_BN_ONLY = False

# CONFIG.TRAIN.VAL_PERCENT = 0.1
CONFIG.TRAIN.VAL_PERCENT = 1

CONFIG.EVAL = edict()
CONFIG.EVAL.NUM_FRAMES = 20

CONFIG.EVAL.CLASSIFICATION_FRACTIONS = [0.1, 0.5, 1.0]
CONFIG.EVAL.KENDALLS_TAU_DISTANCE = "sqeuclidean"  # cosine or sqeuclidean

# DTW Alignment Parameters
CONFIG.DTWALIGNMENT = edict()
CONFIG.DTWALIGNMENT.EMBEDDING_SIZE = 128
CONFIG.DTWALIGNMENT.SDTW_GAMMA = 0.1
CONFIG.DTWALIGNMENT.SDTW_NORMALIZE = False

CONFIG.LOSSES = edict()
CONFIG.LOSSES.IDM_IDX_MARGIN = 2.0
CONFIG.LOSSES.ALPHA = 0.5
CONFIG.LOSSES.SIGMA = 10  # window size
CONFIG.LOSSES.L2_NORMALIZE = True

# TCC Parameters
CONFIG.TCC = edict()
CONFIG.TCC.EMBEDDING_SIZE = 128
CONFIG.TCC.CYCLE_LENGTH = 2
CONFIG.TCC.LABEL_SMOOTHING = 0.1
CONFIG.TCC.SOFTMAX_TEMPERATURE = 0.1
CONFIG.TCC.LOSS_TYPE = "regression_mse_var"
CONFIG.TCC.NORMALIZE_INDICES = True
CONFIG.TCC.VARIANCE_LAMBDA = 0.001
CONFIG.TCC.FRACTION = 1.0
CONFIG.TCC.HUBER_DELTA = 0.1
CONFIG.TCC.SIMILARITY_TYPE = "l2"  # l2, cosine

CONFIG.TCC.TCC_LAMBDA = 1.0

CONFIG.DATA = edict()

CONFIG.DATA.IMAGE_SIZE = 224  # For ResNet50

CONFIG.DATA.SHUFFLE_QUEUE_SIZE = 0
CONFIG.DATA.NUM_PREFETCH_BATCHES = 1
CONFIG.DATA.RANDOM_OFFSET = 1
CONFIG.DATA.FRAME_STRIDE = 16
CONFIG.DATA.SAMPLING_STRATEGY = (
    "segment_uniform"  # offset_uniform, stride, all, segment_uniform
)
CONFIG.DATA.NUM_CONTEXT = 2  # number of frames that will be embedded jointly,
CONFIG.DATA.CONTEXT_STRIDE = 15  # stride between context frames

CONFIG.DATA.FRAME_LABELS = True
CONFIG.DATA.PER_DATASET_FRACTION = 1.0  # Use 0 to use only one sample.
CONFIG.DATA.PER_CLASS = False

# stride of frames while embedding a video during evaluation.
CONFIG.DATA.SAMPLE_ALL_STRIDE = 1

# CONFIG.DATA.TCN = CONFIG.TCN
CONFIG.DATA.WORKERS = 30

# ******************************************************************************
# Augmentation params
# ******************************************************************************
CONFIG.AUGMENTATION = edict()
CONFIG.AUGMENTATION.RANDOM_FLIP = True
CONFIG.AUGMENTATION.RANDOM_CROP = False
CONFIG.AUGMENTATION.BRIGHTNESS_DELTA = 32.0 / 255  # 0 to turn off
CONFIG.AUGMENTATION.CONTRAST_DELTA = 0.5  # 0 to turn off
CONFIG.AUGMENTATION.HUE_DELTA = 0.0  # 0 to turn off
CONFIG.AUGMENTATION.SATURATION_DELTA = 0.0  # 0 to turn off


# ******************************************************************************
# Penn Action Dataset name(added by 0take in 20231127)
# ******************************************************************************
CONFIG.PENN_ACTION = edict()
CONFIG.PENN_ACTION.NAME = None
