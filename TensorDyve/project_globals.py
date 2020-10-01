'''

    Replace all hardcoded values used in more than one place throught the project with 
    variables defined here, even the ones found in the hand crafted config.json

'''

from platform import platform

BEST_METRICS_BUFFER_NAME = 'metrics.json'

BEST_TRAIN_IOU_KEY = 'train_iou'
BEST_EVAL_IOU_KEY = 'eval_iou'
BEST_TRAIN_F1_SCORE_KEY = 'train_f1_score'
BEST_EVAL_F1_SCORE_KEY = 'eval_f1_score'

BEST_TRAIN_ACC_KEY = 'train_acc' 
BEST_EVAL_ACC_KEY = 'eval_acc'

BEST_CKPT_DIR_NAME = 'best_ckpts'

MODEL_INPUT_WIDTH_KEY = 'MODEL_INPUT_WIDTH'
MODEL_INPUT_HEIGHT_KEY = 'MODEL_INPUT_HEIGHT'
MODEL_INPUT_CHANNELS_KEY = 'MODEL_INPUT_CHANNELS'

ALLOWED_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
if platform()[0] == 'W':
    ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

BUTS_DIR_NAME = 'BUTS'

JSON_ANNOTATION_DIR = 'ForTrainingAnnotations'
JSON_ANNOT_KEY = 'Annotations'
JSON_LABEL_KEY = 'Label'
JSON_TOP_LEFT_KEY = 'PointTopLeft'
JSON_TOP_RIGHT_KEY = 'PointTopRight'
JSON_BOTTOM_LEFT_KEY = 'PointBottomLeft'
JSON_BOTTOM_RIGHT_KEY = 'PointBottomRight'
JSON_IS_BAD_IMAGE_KEY = 'IsBadImage'

TFR_IMPLICIT_DIR_NAME = 'TFRECORDS_DATASET'

# Utility MODEs and TASKs

TRAIN = 'train'
PREDICT = 'predict'
EXPORT = 'export'

CLS = 'cls'
DTT = 'dtt'

INPUT_OP_NAME = 'input'
OUTPUT_OP_NAME = 'prediction'

# Sample dir expected content

MODEL_FILE_NAME = 'model.py'
CONFIG_FILE_NAME = 'config.json'

# TFR feature keys for serialization

TFR_SAMPLE_KEY = 'sample'
TFR_GROUND_TRUTH_KEY = 'ground_truth'
TFR_SHAPE_KEY = 'image_shape'

# TFR dir expected content

TRAIN_TFR_NAME = 'train.tfr'
EVAL_TFR_NAME = 'eval.tfr'
METADATA_FILE_NAME = 'metadata.json'

# TFR dataset metadata json keys

NUM_CLASSES_KEY = 'NUM_CLASSES'
NUM_TRAIN_EXAMPLES_KEY = 'NUM_TRAIN_EXAMPLES'
NUM_EVAL_EXAMPLES_KEY = 'NUM_EVAL_EXAMPLES'
ANNOTATION_DS_FLAG_KEY = 'ANNOTATION_DS_FLAG'
TF_VERSION_KEY = 'TF_VERSION'
CLASS_NAMES_KEY = 'CLASS_NAMES'


# ARG PARSER min max ranges

TRAIN_DS_PERCENT_MIN = 0.1
TRAIN_DS_PERCENT_MAX = 0.99

MEM_THRESHOLD_PERCENT_MIN = 0
MEM_THRESHOLD_PERCENT_MAX = 90.

# ARG PARSER STRING OP ARGUMENT KEYWORDS, ignore the grouping, it's just random

MODE_ARG = 'mode'

SRC_DIR_ARG, DST_DIR_ARG = ['pred_src_dir', 'pred_dst_dir' ]
TASK_ARG, CONFIG_ARG, DUMP_ARG, TFR_DEST_DIR_ARG = [ 'task', 'config', 'dump', 'tfr_dest_dir' ]
IMG_DATASET_ARG, TRAIN_IMG_DATASET_ARG, EVAL_IMG_DATASET_ARG = ['img_dataset', 'train_img_dataset', 'eval_img_dataset']
PERCENT_ARG, THRESHOLD_ARG, TFR_DATASET_ARG, NUM_TFR_SHARDS_ARG = ['percent', 'threshold', 'tfr_dataset', 'num_shards']
EXPORT_MODEL_DIR_ARG, EXPORT_MODEL_FILE_NAME_ARG = ['export_model_dir', 'export_model_file_name']
#hide metadata graph from unauthorized user (for cloud training)
HIDE_META_ARG = 'hm' 
TOCO_COMPATIBLE_ARG = 'toco_compat'