from TensorDyve.utils import crop_image 

import tensorflow as tf
import numpy as np

from cv2 import rectangle, putText, imshow, waitKey, FONT_HERSHEY_SIMPLEX, resize, BORDER_CONSTANT, copyMakeBorder, INTER_LINEAR
from random import randint

from keras.backend import tf as ktf

from platform import platform

'''

    Replace all hardcoded values used in more than one place throught the project with 
    variables defined here, even the ones found in the hand crafted config.json

'''

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

TRAIN_TFR_NAME = 'TRAIN.tfr'
EVAL_TFR_NAME = 'EVALUATION.tfr'
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
TASK_ARG, CONFIG_ARG, DUMP_ARG = [ 'task', 'config', 'dump' ]
IMG_DATASET_ARG, COLOR_ARG, EXTENSION_ARG = ['img_dataset', 'color', 'extension']
PERCENT_ARG, THRESHOLD_ARG, TFR_DATASET_ARG = ['percent', 'threshold', 'tfr_dataset']
EXPORT_MODEL_DIR_ARG, EXPORT_MODEL_FILE_NAME_ARG = ['export_model_dir', 'export_model_file_name']


CROP = True

annot_flag = False

features = { 
            
            TFR_SHAPE_KEY          : tf.FixedLenFeature([], tf.string), 
            TFR_SAMPLE_KEY         : tf.FixedLenFeature([], tf.string),
            TFR_GROUND_TRUTH_KEY  : tf.FixedLenFeature([], tf.string) if annot_flag else tf.FixedLenFeature([], tf.int64) 
                
        }

TFR_FILE = '/home/catalinh/ml/workspace/biomet/models/cls/orange_juice_models/asd/TFRECORDS_DATASET/TRAIN.tfr'

num_classes = 1

resize_shape = (320, 320)

yolo_spec = {
    
          "OBJ_THRESHOLD": 0.5,
          "NMS_THRESHOLD": 0.3,
          "IOU_THRESHOLD": 0.4,
          "ANCHORS": [
               0.26,
               0.97,
               0.82,
               0.84,
               0.97,
               0.26
          ],
          "NO_OBJECT_SCALE": 1.0,
          "OBJECT_SCALE": 5.0,
          "COORD_SCALE": 1.0,
          "CLASS_SCALE": 1.0,
          "CLASS_WEIGHTS": [
               1.0
          ],
          "WARM_UP_BATCHES": 0,
          "TRUE_BOX_BUFFER": 50,
          "GRID_W": 5,
          "GRID_H": 5
     }


def decode_image_shape(raw_image_shape):
        
    _shape_dtype = tf.int32
        
    return tf.decode_raw(raw_image_shape, _shape_dtype)

def build_image(raw_image, image_shape):
        
    _image_dtype = tf.uint8
    image = tf.decode_raw(raw_image, _image_dtype)
      
    return tf.reshape(image, image_shape) # reshaping flattened version of the image
    
def build_ground_truth(raw_ground_truth):
       
    _annot_flag = False 
    
    if _annot_flag:
            
        '''
        
        annotations for one image --> [ [xmin, xmax, ymin, ymax, cls_index], [xmin, xmax, ymin, ymax, cls_index], [xmin, xmax, ymin, ymax, cls_index], ... ]
            
        * after decode we get the flattened array, of which the last element is the number of annotations
        * poping last element and using it to reshape the remaining array to [num_annotations, elements_per_annotation]
                            
        '''
        
        annotations = tf.decode_raw(raw_ground_truth, tf.float32)
        num_elements = tf.size(annotations)
        num_annotations = annotations[ num_elements - 1 ]
        ground_truth = tf.slice(annotations, [0], [ num_elements - 1 ])

        ground_truth = tf.reshape(ground_truth, [ num_annotations, tf.cast(6, tf.int32) ] )
            
    else :
        
        ground_truth = tf.cast(raw_ground_truth, tf.int64)
            
        ground_truth = tf.one_hot(
            
                indices = ground_truth,
                depth = 7,
                on_value = tf.cast(1, tf.float32),
                off_value = tf.cast(0, tf.float32),
                dtype = tf.float32
                    
        )
        
        return ground_truth
    
def _resize_image_wrapper():
    
    def _resize_image(image, ground_truth):
            
        #print(self._model_input_width, self._model_input_height)
        return resize(image, (640, 640), interpolation = INTER_LINEAR), ground_truth
        
    return _resize_image    

def _build_resize_py_func(image, ground_truth):
        
        py_func_op = tf.py_func(func = _resize_image_wrapper(), inp=[ image, ground_truth ], Tout=(np.uint8, np.float32))

        py_func_op[0].set_shape([320,320,3]) # requires expliciticity

        return py_func_op    

def _deserialize(serialized):

    global features

    parsed_example = tf.parse_single_example(serialized, features)
        
    image_shape   = decode_image_shape( parsed_example[ TFR_SHAPE_KEY ] )
    image         = build_image( parsed_example[ TFR_SAMPLE_KEY ], image_shape)
    ground_truth = build_ground_truth(parsed_example[ TFR_GROUND_TRUTH_KEY ])
        
        
    return tf.image.resize_image_with_crop_or_pad(image, image_shape[1], image_shape[0]), ground_truth

def _build_crop_image_py_func(image, annotations):

    py_func_op = tf.py_func( func = crop_image, inp = [ image, annotations, resize_shape ], Tout = (np.uint8, np.int32))
    
    py_func_op[0].set_shape( image.get_shape() )
    
    return py_func_op

dataset = tf.data.TFRecordDataset(TFR_FILE)

dataset = dataset.map( lambda serialized: _deserialize(serialized) )

#dataset = dataset.map( lambda img, gt: _build_resize_py_func( img, gt ), 8 )

#dataset = dataset.map( 
        
#            lambda img, gt: ( tf.image.resize_images(img, (640, 320)), gt), 
#            8 
            
#        )


dataset = dataset.repeat(1)

iterator = dataset.make_one_shot_iterator()

CLASS_NAMES = [
                    "42-5399-051-03",
                    "42-5399-052-07",
                    "00-5983-070-12",
                    "00-5983-070-10",
                    "42-5399-052-03",
                    "00-5983-070-14",
                    "42-5399-051-07"
          ]
with tf.Session() as sess:

    while True:

        image, ground_thruth = iterator.get_next()

        img, gt = sess.run( [ image, ground_thruth ] )

        if not CROP:
            
            for _gt in gt:
                
                rectangle(img, (_gt[0], _gt[1]), (_gt[2], _gt[3]), (0, 255, 0), 2)

        imshow(CLASS_NAMES[np.argmax(gt)], img.astype(np.uint8))
        waitKey(0)

