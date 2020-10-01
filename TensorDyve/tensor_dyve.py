import tensorflow as tf
import logging

from os.path import join, basename, dirname
from argparse import ArgumentParser
from json import load, dump

from platform import platform

from sys import path as PYTHONPATH
from sys import float_info

from .custom_exceptions import MemThresholdReached

from .dataset_utils import ImgDataset, TFRGenerator

from .estimator_prep import EstimatorInputPipe, ClassifierEstimator, YoloEstimator

from .project_globals import ALLOWED_EXTENSIONS, MODEL_INPUT_WIDTH_KEY, \
    MODEL_INPUT_HEIGHT_KEY, MODEL_INPUT_CHANNELS_KEY, MODE_ARG, TASK_ARG, CONFIG_ARG, \
    DUMP_ARG, TFR_DEST_DIR_ARG, HIDE_META_ARG, IMG_DATASET_ARG, TRAIN_IMG_DATASET_ARG, EVAL_IMG_DATASET_ARG,\
    PERCENT_ARG, THRESHOLD_ARG, TFR_DATASET_ARG, \
    EXPORT_MODEL_DIR_ARG, EXPORT_MODEL_FILE_NAME_ARG, SRC_DIR_ARG, DST_DIR_ARG, \
    TRAIN, PREDICT, EXPORT, CLS, DTT, TRAIN_DS_PERCENT_MIN, TRAIN_DS_PERCENT_MAX, \
    MEM_THRESHOLD_PERCENT_MIN, MEM_THRESHOLD_PERCENT_MAX, TRAIN_TFR_NAME, \
    EVAL_TFR_NAME, CONFIG_FILE_NAME, MODEL_FILE_NAME, METADATA_FILE_NAME, INPUT_OP_NAME, TOCO_COMPATIBLE_ARG, NUM_TFR_SHARDS_ARG

from .utils import build_estimator_model_function, update_config, \
    check_args_validity, export_inference_model

parser = ArgumentParser(description = 'Note that one has to either specify ' + IMG_DATASET_ARG + ' or the pair ' + TRAIN_IMG_DATASET_ARG + ', ' + EVAL_IMG_DATASET_ARG)

parser.add_argument(
    
    MODE_ARG, 
    choices=[TRAIN, PREDICT, EXPORT], 
    help='mode of operation for the utility.'
    
)

parser.add_argument(
    
    '--' + TASK_ARG, 
    choices=[CLS, DTT], 
    help='wether to work with a classifier or yolo detector'

)
parser.add_argument(
    
    '--' + CONFIG_ARG, 
    help='path to configuration root directory, where config.json and model.py is to be found'

)

parser.add_argument(
    
    '--' + DUMP_ARG, 
    help='path to model checkpoints and tensorboard loging dump directory'

)

parser.add_argument(
    
    '--' + IMG_DATASET_ARG, 
    help='Path to image dataset root dir '

)
parser.add_argument(
    
    '--' + TRAIN_IMG_DATASET_ARG, 
    help='Path to train image dataset root dir'

)

parser.add_argument(
    
    '--' + EVAL_IMG_DATASET_ARG, 
    help='Path to eval image dataset root dir '

)
parser.add_argument(
                    
    '--' + PERCENT_ARG,
    type=float,
    help='Train dataset percentage, float between ' + str(TRAIN_DS_PERCENT_MIN) + ' and ' + str(TRAIN_DS_PERCENT_MAX)
                
)
                
parser.add_argument(
                    
    '--' + THRESHOLD_ARG,
    type=float,
    help='Memory threshold cap for loading dataset, float between ' + str(MEM_THRESHOLD_PERCENT_MIN) + ' and ' + str(MEM_THRESHOLD_PERCENT_MAX)
    
)

parser.add_argument(
    
    '--' + TFR_DATASET_ARG, 
    help='Path to tfr dataset root dir'
    
)

parser.add_argument(
    
    '--' + NUM_TFR_SHARDS_ARG,
    type=int,
    help='number of tfr shards for train and test set to be split into'
    
)

parser.add_argument(
    
    '--' + SRC_DIR_ARG, 
    help='Source directory for images that are to be used for ' + PREDICT

)

parser.add_argument(
    
    '--' + DST_DIR_ARG, 
    help='Destination directory for ' + PREDICT

)

parser.add_argument(
    
    '--' + EXPORT_MODEL_DIR_ARG, 
    help='Destination directory where the inference model will be dumped'

)
parser.add_argument(
    
    '--' + EXPORT_MODEL_FILE_NAME_ARG, 
    help='String ending in .pb'

)

parser.add_argument(
    
    '--' + TFR_DEST_DIR_ARG,
    required=False,
    help='path to the root folder where TF record files are saved (default is dump)'
)

parser.add_argument(
    
    '-' + HIDE_META_ARG,
    required=False,
    action='store_true'
    #this flag is indended for automatic training on cloud
    #do not help the unauthorized user
    #,help='hide graph metadata'
)

parser.add_argument(

    '-' + TOCO_COMPATIBLE_ARG,
    required = False,
    action = 'store_true',
    help = 'wether to export frozen inference graph protobuf compatible with the TOCO converter utility'

)

arguments = vars (parser.parse_args())

checked_arguments = check_args_validity(arguments)

( MODE, TASK, CONFIG_DIR, DUMP_DIR, TFR_DEST_DIR, IMG_DATASET, PERCENT, TRAIN_IMG_DATASET, EVAL_IMG_DATASET,
  THRESHOLD, TFR_DATASET, SRC_DIR, DST_DIR, 
  EXPORT_MODEL_DIR, EXPORT_MODEL_FILE_NAME, HIDE_META, TOCO_COMPATIBLE, TFR_NUM_SHARDS ) = checked_arguments 


CONFIG_FILE = join(CONFIG_DIR, CONFIG_FILE_NAME)

'''

    CALL config_validity_check(CONFIG_FILE) and get the dict

'''

with open(CONFIG_FILE, 'r') as handler:
        
    CONFIG = load(handler)

MODEL_INPUT_WIDTH = CONFIG[MODEL_INPUT_WIDTH_KEY]
MODEL_INPUT_HEIGHT = CONFIG[MODEL_INPUT_HEIGHT_KEY]
MODEL_INPUT_CHANNELS = CONFIG[MODEL_INPUT_CHANNELS_KEY]

if MODEL_INPUT_CHANNELS not in [1, 3]: 
    
    msg = 'Currently only grayscale and rgb images are to be used'
    raise ValueError(msg)

COLOR = MODEL_INPUT_CHANNELS == 3

YOLO_SPEC = CONFIG.get('YOLO_SPEC', None)
TRAIN_CONFIG = CONFIG['TRAIN_CONFIG']

BATCHING_SPEC       = TRAIN_CONFIG['BATCHING']

QUANT_DELAY         = TRAIN_CONFIG.get('QUANTIZE_DELAY', -1)
MAX_EPOCHS          = TRAIN_CONFIG.get('MAX_EPOCHS', -1)
MAX_BATCHES         = TRAIN_CONFIG.get('MAX_BATCHES', -1)
MAX_TOLERANCE_COUNT = TRAIN_CONFIG.get('MAX_TOLERANCE_COUNT', -1)

PYTHONPATH.insert(0, CONFIG_DIR)

from model import *

MODEL_INPUT_SHAPE = [ MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS ]

USER_DEFINED_MODEL_FN = locals()['model_function']
USER_DEFINED_AUG_PIPE = locals()['AUG_PIPE']

if MODE == TRAIN:

    '''
    
        Get dataset metadata
    
    '''

    if TFR_DATASET is not None:
        
        with open( join( TFR_DATASET, METADATA_FILE_NAME ), 'r' ) as handler:
            
            dataset_metadata = load( handler )
        
        NUM_TRAIN_EXAMPLES = dataset_metadata['NUM_TRAIN_EXAMPLES']
        NUM_EVAL_EXAMPLES = dataset_metadata['NUM_EVAL_EXAMPLES']
        NUM_CLASSES = dataset_metadata['NUM_CLASSES']
        CLASS_NAMES = dataset_metadata['CLASS_NAMES']
        
        ANNOT_DS_FLAG = dataset_metadata['ANNOTATION_DS_FLAG'] 
         
    else: # I.E. img dataset split or otherwise
        
        print('Preping handle for image dataset ...')
        # some of these values will be None and it's fine, the ImgDataset instance will know how to handle them, yielding a valid dataset handle to work with
        image_dataset = ImgDataset( COLOR, THRESHOLD, IMG_DATASET, PERCENT, TRAIN_IMG_DATASET, EVAL_IMG_DATASET )
    
        NUM_TRAIN_EXAMPLES = image_dataset.get_num_train_examples()
        NUM_EVAL_EXAMPLES = image_dataset.get_num_eval_examples()
        NUM_CLASSES = image_dataset.get_num_classes()
        CLASS_NAMES = image_dataset.get_class_names()
        
        ANNOT_DS_FLAG = image_dataset.is_annotation_dataset()
        
        
    TRAIN_INFO = {  
                        
        'TASK'               : TASK,
        'NUM_TRAIN_EXAMPLES' : NUM_TRAIN_EXAMPLES,
        'NUM_EVAL_EXAMPLES'  : NUM_EVAL_EXAMPLES,
        'NUM_CLASSES'        : NUM_CLASSES,
        'TF_VERSION'         : tf.__version__,
        'CLASS_NAMES'        : CLASS_NAMES
                                                  
    }                     
                
    pipe_builder = EstimatorInputPipe(
        
        TASK == CLS, 
        ANNOT_DS_FLAG, 
        USER_DEFINED_AUG_PIPE, 
        BATCHING_SPEC, 
        NUM_CLASSES, 
        YOLO_SPEC, 
        MODEL_INPUT_SHAPE
    
    )

    if TFR_DATASET is not None:
        
        def train_input_pipe():
            
            return pipe_builder.build_tfr_pipe( join( TFR_DATASET, '*' + TRAIN_TFR_NAME ), train_flag = True )
        
        def eval_input_pipe():
    
            return pipe_builder.build_tfr_pipe( join( TFR_DATASET, '*' + EVAL_TFR_NAME ), train_flag = False )

        print('Finished building input pipes for both training and evaluation dataset found in tfr serialized form on disk')
    
    else: # I.E IMG Dataset
        
        try:
            
            train_samples, train_gts, eval_samples, eval_gts = image_dataset.load_in_memory()
            
            def train_input_pipe():
                
                return pipe_builder.build_mem_pipe( train_samples, train_gts, train_flag = True )
            
            def eval_input_pipe():
                
                return pipe_builder.build_mem_pipe( eval_samples, eval_gts, train_flag = False )
            
            print('Finished building input pipes for both training and evaluation dataset found in memory')
            
        except MemThresholdReached as mtr:
            
            print(mtr)
            
            converter = TFRGenerator( image_dataset, num_tfr_shards = TFR_NUM_SHARDS, tfr_destination_directory = TFR_DEST_DIR ) # added dst dir since user can specify 
            
            path_to_tfr_dataset = converter.convert() # images are to be serialezd together with their shapes and ground truths regardless of the task ahead
            
            def train_input_pipe():
                
                return pipe_builder.build_tfr_pipe( join( path_to_tfr_dataset, '*' + TRAIN_TFR_NAME ), train_flag = True )
            
            def eval_input_pipe():
            
                return pipe_builder.build_tfr_pipe( join( path_to_tfr_dataset, '*' + EVAL_TFR_NAME ), train_flag = False )
    
            print('Finished building input pipes for both training and evaluation dataset that was just serialized to disk')
        
    NUM_BATCHES_IN_EPOCH = int(NUM_TRAIN_EXAMPLES / BATCHING_SPEC['BATCH_SIZE'])

    if MAX_EPOCHS > 0:
        if MAX_BATCHES > 0:
            MAX_BATCHES = min(MAX_BATCHES, MAX_EPOCHS * NUM_BATCHES_IN_EPOCH)
        else:
            MAX_BATCHES = MAX_EPOCHS * NUM_BATCHES_IN_EPOCH

    TRAIN_STEPS = int( ( NUM_TRAIN_EXAMPLES / BATCHING_SPEC['BATCH_SIZE'] ) * TRAIN_CONFIG['LOOP_OVER_TRAIN_SET_COUNT'] )
    
    update_config(CONFIG_FILE, { 'TRAIN_INFO': TRAIN_INFO } )
 
else:
    
    TRAIN_STEPS = None
    
    try:
    
        TRAIN_INFO = CONFIG['TRAIN_INFO']
    
    except KeyError:
    
        msg = PREDICT + ' mode requires a config file that has the TRAIN_INFO key present. The file provided does not seem to have the signature of one used in training, which would have had the relevant info needed for predictions'
        raise ValueError(msg)

### Variables modified by the session run hook. Based on the the utility will stop
best_train_acc = 0
best_eval_acc = 0

best_train_iou = 0
best_train_f1_score = 0
best_eval_iou = 0
best_eval_f1_score = 0

def main():

    SESSION_RUN_CONFIG = None

    _os = platform()
    if _os[0] == 'W' :  # for Windows

        SESSION_RUN_CONFIG = tf.ConfigProto()
        SESSION_RUN_CONFIG.gpu_options.allow_growth = True

    RUN_CONFIG = tf.estimator.RunConfig(
        
        save_summary_steps     = None,
        log_step_count_steps   = None,
        keep_checkpoint_max    = 3, # the tf.train.saver that the tf estimator instance builds will be fetched and used to manually do checkpointing when desired
        save_checkpoints_steps = None if HIDE_META else TRAIN_STEPS,
        session_config         = SESSION_RUN_CONFIG
                        
    )
    
    ESTIMATOR_PARAMS = { 
              
        'YOLO_SPEC'            : YOLO_SPEC,
        'MODEL_FUNCTION'       : USER_DEFINED_MODEL_FN,
        'TRAIN_INFO'           : TRAIN_INFO,
        'TASK'                 : TASK,
        'MODE'                 : MODE,
        'MODEL_INPUT_WIDTH'    : MODEL_INPUT_WIDTH,
        'MODEL_INPUT_HEIGHT'   : MODEL_INPUT_HEIGHT,
        'MODEL_INPUT_CHANNELS' : MODEL_INPUT_CHANNELS,
        'CKPT_DIR'             : DUMP_DIR,
        'QUANTIZE_DELAY'       : QUANT_DELAY,
        'MAX_TOLERANCE_COUNT'  : MAX_TOLERANCE_COUNT,
        'HIDE_META'            : HIDE_META
              
    }

    if MODE == EXPORT:
       
        # TODO: see wether tf estimator instance can initialize from checkpoint without invoking train, predict or eval
        with tf.Session() as sess:
            
            # since there is no estimator instance to build the graph properly, leftovers like global step has to be defined here since the user can define request the global step op within model.py
            tf.train.global_step( sess, tf.constant( 0, name = "global_step", dtype = tf.int32) ) 
            
            _ = build_estimator_model_function(
                
                features = tf.placeholder( shape = [None, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS], dtype = tf.float32, name = INPUT_OP_NAME),
                labels = tf.constant( [1] * TRAIN_INFO['NUM_CLASSES'], dtype = tf.float32), # dummy values just to build the graph, currently it does not hold for a detector model, it's just a silly hack to suite urgent needs
                mode = tf.estimator.ModeKeys.PREDICT, 
                params = ESTIMATOR_PARAMS
                
            )
           
        export_inference_model(DUMP_DIR, EXPORT_MODEL_DIR, EXPORT_MODEL_FILE_NAME, TOCO_COMPATIBLE)
        exit() # no need to go any further
    
    if TASK == DTT:

        estimator = YoloEstimator(
        
            model_fn  = build_estimator_model_function,
            params    = ESTIMATOR_PARAMS, # a way to sift any sort of info into the model_fn, called by the estimator api
            model_dir = DUMP_DIR,
            config    = RUN_CONFIG 
                                        
        )
        
    else: # I.E. CLS
           
        estimator = ClassifierEstimator(
        
            model_fn  = build_estimator_model_function,
            params    = ESTIMATOR_PARAMS,
            model_dir = DUMP_DIR,
            config    = RUN_CONFIG 
                                        
        )
   
    if MODE == TRAIN:
        
        '''
        
            TODO:
            
                relocate early stopping mechanism in best checkpointing hook
                here there will be an exception to be caught in order to stop the loop
        
        '''
        
        while True:
            
            estimator.train( input_fn = train_input_pipe, steps = TRAIN_STEPS )
            evaluation_dict = estimator.evaluate( input_fn = eval_input_pipe )
                
            if MAX_BATCHES > 0:
                global_step = evaluation_dict['global_step']
                if global_step >= MAX_BATCHES:
                    print('User defined step limit has been reached:', global_step, '. Training finished.')
                    break

            if TASK == DTT:

                if best_train_iou == 1 and best_train_f1_score == 1 and best_eval_iou == 1 and best_eval_f1_score == 1:
                    
                    break
                
            else:

                if best_train_acc == 1 and best_eval_acc == 1:
                    
                    break

    else: # I.E. PREDICT, since EXPORT does not reach this far
        
        estimator.predict_wrapper(SRC_DIR, DST_DIR)

