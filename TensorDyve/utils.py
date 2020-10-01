import tensorflow as tf
import numpy as np

from os import listdir
from os.path import isdir, join

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference # seems like you can't find this in the tf namespace, had to scour for it

from cv2 import resize, copyMakeBorder, BORDER_CONSTANT
from glob import glob

from json import load, dump
from random import randint
from re import split

from .custom_exceptions import MissingModelFile, MissingConfigFile, \
    MissingTrainTFR, MissingEvalTFR, ModelDatasetMissmatch, MissingMetaDataFile

from .project_globals import INPUT_OP_NAME, OUTPUT_OP_NAME, MODE_ARG, TASK_ARG, \
    CONFIG_ARG, DUMP_ARG, TFR_DEST_DIR_ARG, HIDE_META_ARG, IMG_DATASET_ARG, TRAIN_IMG_DATASET_ARG, EVAL_IMG_DATASET_ARG, \
    PERCENT_ARG, THRESHOLD_ARG, TFR_DATASET_ARG, \
    EXPORT_MODEL_DIR_ARG, EXPORT_MODEL_FILE_NAME_ARG, SRC_DIR_ARG, DST_DIR_ARG, \
    TRAIN, PREDICT, EXPORT, CLS, DTT, TRAIN_DS_PERCENT_MIN, TRAIN_DS_PERCENT_MAX, \
    MEM_THRESHOLD_PERCENT_MIN, MEM_THRESHOLD_PERCENT_MAX, TRAIN_TFR_NAME, \
    EVAL_TFR_NAME, METADATA_FILE_NAME, CONFIG_FILE_NAME, MODEL_FILE_NAME, TOCO_COMPATIBLE_ARG, NUM_TFR_SHARDS_ARG
    

from .checkpointing import ClassifierSessionRunHook, DetectorSessionRunHook

def LOG(message):
    
    print(message)


def sigmoid(x):
        
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
        
    x = x - np.max(x)
    
    if np.min(x) < t:
        
        x = x / np.min(x) * t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

def get_absolute_bbox(relative_bbox, image_w, image_h):

    xmin = int(relative_bbox[0] * image_w)
    ymin = int(relative_bbox[1] * image_h)
    xmax = int(relative_bbox[2] * image_w)
    ymax = int(relative_bbox[3] * image_h)
    
    obj_prob = relative_bbox[4]
    
    cls_index = np.argmax(relative_bbox[ 5: ] )
    
    return [xmin, ymin, xmax, ymax, obj_prob, cls_index]

def convert_grid_relative_bboxes_from_netout(net_out, num_classes, yolo_spec):
        
        grid_h, grid_w, nb_box = net_out.shape[: 3 ]

        boxes = []
        
        net_out[ ..., 4 ] = sigmoid(net_out[..., 4])
        net_out[ ..., 5: ] = softmax(net_out[..., 5:])
        
        for row in range(grid_h):
            
            for col in range(grid_w):
                
                for b in range(nb_box):
                    
                    classes = net_out[ row, col, b, 5: ]
                    
                    x, y, w, h = net_out[ row, col, b, :4]
    
                    x = (col + sigmoid(x)) / grid_w 
                    y = (row + sigmoid(y)) / grid_h
                    
                    w = yolo_spec['ANCHORS'][ 2 * b + 0 ] * np.exp(w) / grid_w
                    h = yolo_spec['ANCHORS'][ 2 * b + 1 ] * np.exp(h) / grid_h
                    
                    confidence = net_out[ row, col, b, 4 ]
                    
                    bbox = np.append( [ x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence ], classes).astype(np.float32)              
                    boxes.append(bbox)
                    
        return boxes

def compute_detector_metrics_wrapper(num_classes, yolo_spec, width, height):

    def compute_detector_metrics(net_out_batch, ground_truths_batch):

        batch_avg_ious = []
        batch_true_pos, batch_false_pos, batch_false_neg = 0, 0, 0
        
        for net_out, ground_truths in zip(net_out_batch, ground_truths_batch):

            xy_rel_bboxes = convert_grid_relative_bboxes_from_netout(net_out, num_classes, yolo_spec)
    
            if np.sum(ground_truths[0]) == 0: # possible negative example, ground_truths == [ [0, 0, 0, 0, 0, 0] ]
                
                continue # we skip this iteration since we don't want to compute avg iou between said prediction and the dummy values for this negative example label

            abs_bboxes = []
            for xy_rel_bbox in xy_rel_bboxes:

                abs_bbox = get_absolute_bbox(xy_rel_bbox, width, height)
    
                abs_bboxes.append( abs_bbox )
                
            all_ground_truth_ious = []
            
            for ground_truth in ground_truths:

                ious = []
                is_false_neg = True
    
                for abs_bbox in abs_bboxes:

                    iou = bbox_iou(ground_truth, abs_bbox)
                    ious.append(iou)

                    if abs_bbox[4] >= yolo_spec['OBJ_THRESHOLD']:
                        
                        is_false_neg = False
                        
                        if iou >= yolo_spec['IOU_THRESHOLD'] and abs_bbox[5] == ground_truth[5]:
                            
                            batch_true_pos += 1
                        
                        else:
                            
                            batch_false_pos += 1
                        
                if is_false_neg : batch_false_neg += 1
                
                all_ground_truth_ious.append( max(ious) )

            example_avg_iou = np.mean(all_ground_truth_ious, dtype = np.float32)
            
            batch_avg_ious.append( example_avg_iou ) # we end up with a list of avg iou values, each value beeing associated with one particular example
        
        return np.asarray(batch_avg_ious, dtype = np.float32), batch_true_pos, batch_false_pos, batch_false_neg
    
    return compute_detector_metrics

def get_detector_metrics(net_out, ground_truths, num_classes, yolo_spec, width, height):
    
    py_func_op = tf.py_func( 
                            
            func = compute_detector_metrics_wrapper( 
                                                   
                        num_classes, 
                        yolo_spec, 
                        width, 
                        height ), 
                            
                    inp = [ net_out, ground_truths ], 
                    Tout = (  tf.float32, tf.int64, tf.int64, tf.int64 ) 
    )
    
    batch_mean_iou, batch_true_pos, batch_false_pos, batch_false_neg = py_func_op
    
    #TODO: make the pyfunc return all the detector metrics so that they can be unpacked here, and proccessed accordingly in order to return the update ops for streaming metrics
    value, update_op = tf.metrics.mean(batch_mean_iou)
    
    return value, update_op, batch_true_pos, batch_false_pos, batch_false_neg

def export_inference_model(ckpt_dir, model_dir, model_name, toco_compatible):

    graph = tf.get_default_graph()

    with tf.Session(graph=graph) as sess:
        
        saver = tf.train.Saver()

        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir) )

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            [ OUTPUT_OP_NAME ]
        ) 

        inference_graph_def = optimize_for_inference(

            frozen_graph_def,
            input_node_names = [INPUT_OP_NAME ],
            output_node_names = [ OUTPUT_OP_NAME ],
            placeholder_type_enum=tf.float32.as_datatype_enum,
            toco_compatible=toco_compatible  # look into what this param is about
    
        )

        inference_model_file_name = join(model_dir, model_name)

        with tf.gfile.GFile(inference_model_file_name, "wb") as f:
            f.write(inference_graph_def.SerializeToString())

    print('Inference graph can be found at ', inference_model_file_name)

def crop_image(image, annotations, resize_shape):
    
    '''
    
        Returnes one crop given by annotations[ x ], where x is chosen randomly, 
        and resizes the crop to desired shape, keeping aspect ratio.
    
    '''
    
    if len(annotations) == 0:
        
        print(annotations)
    
    annot_index = randint( 0, len(annotations) - 1 )
    annot = annotations[annot_index]
    
    xmin = annot[0]
    ymin = annot[1]
    xmax = annot[2]
    ymax = annot[3]
    cls_index = annot[5]
    
    crop_w = xmax - xmin
    crop_h = ymax - ymin
    
    center_x = xmin + crop_w // 2
    center_y = ymin + crop_h // 2
    
    model_input_w = resize_shape[1]
    model_input_h = resize_shape[0]
    
    image_w = image.shape[1]
    image_h = image.shape[0]
    
    new_crop_w, new_crop_h = get_desired_wh(crop_w, crop_h, model_input_w / model_input_h )
    
    new_xmin = center_x - new_crop_w // 2
    new_xmax = center_x + new_crop_w // 2
    
    new_ymin = center_y - new_crop_h // 2
    new_ymax = center_y + new_crop_h // 2
    
    pad_left = pad_top = pad_right = pad_bottom = 0
    
    if new_xmin < 0: 
        
        pad_left = abs(new_xmin)
        new_xmin = 0
        
    if new_ymin < 0: 
        
        pad_top = abs(new_ymin)
        new_ymin = 0
    
    bbox_width_reach = new_xmin + new_crop_w
    
    if bbox_width_reach > image_w: pad_right = bbox_width_reach - image_w 
    
    bbox_height_reach = new_ymin + new_crop_h
    
    if bbox_height_reach > image_h: pad_bottom = bbox_height_reach = image_h
    
    image = copyMakeBorder(image, top = pad_top, bottom = pad_bottom, left = pad_left, right = pad_right, borderType = BORDER_CONSTANT, value = 0)
    
    crop = image[ new_ymin: new_ymin + new_crop_h, new_xmin: new_xmin + new_crop_w ]
    crop = resize( crop, (model_input_w, model_input_h) )
    
    return crop, cls_index


def get_desired_wh(in_w, in_h, desirable_ratio):
    
    pad_w = pad_h = 0
    
    if in_w / in_h > desirable_ratio:
        
        pad_h = int( in_w / desirable_ratio - in_h )
        
    else:
        
        pad_w = int( desirable_ratio * in_h - in_w )

    return (in_w + pad_w, in_h + pad_h)

def interval_overlap(interval_a, interval_b):
    
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2, x4) - x3

def bbox_iou(box1, box2):
    
    # bbox --> [xmin, ymin, xmax, ymax, ..]
    
    intersect_w = interval_overlap( [ box1[0], box1[2] ], [ box2[0], box2[2] ] )
    intersect_h = interval_overlap( [ box1[1], box1[3] ], [ box2[1], box2[3] ] )  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union


def check_args_validity(args):
    
    print('Checking argument validity ...')
    
    # doing this because i'd rather unpack everything when returning from this call, main stays cleaner
    
    MODE = args[MODE_ARG]
    TASK = args[TASK_ARG]
    
    CONFIG_DIR = args[CONFIG_ARG]
    DUMP_DIR = args[DUMP_ARG]
    
    IMG_DATASET = args[IMG_DATASET_ARG]
    TRAIN_IMG_DATASET = args[TRAIN_IMG_DATASET_ARG]
    EVAL_IMG_DATASET = args[EVAL_IMG_DATASET_ARG]
    
    TFR_DATASET = args[TFR_DATASET_ARG]
    TFR_DEST_DIR = args[TFR_DEST_DIR_ARG]
    TFR_NUM_SHARDS = args[NUM_TFR_SHARDS_ARG]
    
    PERCENT = args[PERCENT_ARG]
    THRESHOLD = args[THRESHOLD_ARG]
    
    SRC_DIR = args[SRC_DIR_ARG]
    DST_DIR = args[DST_DIR_ARG]
    
    EXPORT_MODEL_DIR = args[EXPORT_MODEL_DIR_ARG]
    EXPORT_MODEL_FILE_NAME = args[EXPORT_MODEL_FILE_NAME_ARG]
    
    TOCO_COMPATIBLE = args[TOCO_COMPATIBLE_ARG]

    HIDE_META = args[HIDE_META_ARG]

    TRAIN_COMBO = { TASK_ARG, CONFIG_ARG, DUMP_ARG }
        
    IMG_DATASET_COMBO       = { IMG_DATASET_ARG, PERCENT_ARG, THRESHOLD_ARG, NUM_TFR_SHARDS_ARG }
    SPLIT_IMG_DATASET_COMBO = { TRAIN_IMG_DATASET_ARG, EVAL_IMG_DATASET_ARG, THRESHOLD_ARG, NUM_TFR_SHARDS_ARG }
    TFR_DATASET_COMBO       = { TFR_DATASET_ARG }

    PREDICT_COMBO = { TASK_ARG, CONFIG_ARG, DUMP_ARG, SRC_DIR_ARG, DST_DIR_ARG }
    EXPORT_COMBO = { TASK_ARG, CONFIG_ARG, DUMP_ARG, EXPORT_MODEL_DIR_ARG, EXPORT_MODEL_FILE_NAME_ARG }

    #checked arguments from all combos
    COMBO_ARGUMENTS = set.union(TRAIN_COMBO, SPLIT_IMG_DATASET_COMBO, IMG_DATASET_COMBO, TFR_DATASET_COMBO, PREDICT_COMBO, EXPORT_COMBO)

    user_arg_combo = set()

    for key, value in args.items():
        
        if not key in COMBO_ARGUMENTS: continue
              
        if value is not None: user_arg_combo.add(key)
        
    split_img_dataset_flag = user_arg_combo == TRAIN_COMBO.union(SPLIT_IMG_DATASET_COMBO)
    img_dataset_flag       = user_arg_combo == TRAIN_COMBO.union(IMG_DATASET_COMBO)
    tfr_dataset_flag       = user_arg_combo == TRAIN_COMBO.union(TFR_DATASET_COMBO)
    
    ########################################
    
    if MODE == TRAIN:

        if tfr_dataset_flag and TFR_DEST_DIR is not None:
            
            err_msg = '{} is redundant if a valid training combo of arguments from a tfr dataset has been provided, better to keep it clean.'.format(TFR_DEST_DIR_ARG)
            raise ValueError(err_msg)

        if ( 
            
                (split_img_dataset_flag and tfr_dataset_flag) or 
                (img_dataset_flag and tfr_dataset_flag) or 
                (img_dataset_flag and split_img_dataset_flag) or 
                (img_dataset_flag and split_img_dataset_flag and tfr_dataset_flag) or
                not ( img_dataset_flag or tfr_dataset_flag or split_img_dataset_flag )
                
        ):
            
            err_msg = ( 'You must provide one of the following combos:\n ' +
                        
                str( TRAIN_COMBO.union( IMG_DATASET_COMBO ) ) + ' for training with an image dataset or\n' +
                str( TRAIN_COMBO.union( SPLIT_IMG_DATASET_COMBO ) ) + ' for training with a split image dataset or\n' +
                str( TRAIN_COMBO.union( TFR_DATASET_COMBO ) ) + ' for training with an already split tfr dataset.' 
                            
            )
            
            raise ValueError(err_msg)
    
    else:
        
        if TFR_DEST_DIR is not None:
            
            err_msg = '{} is redundant if {} is not {}, better to keep it clean.'.format(TFR_DEST_DIR_ARG, MODE_ARG, TRAIN)
            
            raise ValueError(err_msg)

        if HIDE_META:
            
            err_msg = '{} is redundant if {} is different than {}, better to keep it clean.'.format(HIDE_META_ARG, MODE_ARG, TRAIN)
            
            raise ValueError(err_msg)
        
    ########################################

    if MODE == PREDICT:

        if user_arg_combo != PREDICT_COMBO:
            
            err_msg = 'You can only use the following combo with ' + PREDICT + ' mode: ' + str(PREDICT_COMBO) 
            
            raise ValueError(err_msg)
        
        if not isdir(SRC_DIR):
            
            print(SRC_DIR)
            err_msg = 'Predict source directory does not exist'
            
            raise ValueError(err_msg)
            
        if not isdir(DST_DIR):
            
            err_msg = 'Predict destination directory does not exist'
            
            raise ValueError(err_msg)
        
    if MODE == EXPORT:
                                
        if user_arg_combo != EXPORT_COMBO:
            
            raise ValueError('You can only use the followig combo with ' + EXPORT + ' mode: ' + str(EXPORT_COMBO))
    
    if MODE in [ TRAIN, PREDICT ]:
        
        # for EXPORT mode we don't specify config_dir or dump_dir, call to isdir will fail with None type arg
        
        if not isdir(CONFIG_DIR) : raise IOError('Config directory does not exist !')
        if not isdir(DUMP_DIR) : raise IOError('Dump directory does not exist !')
        
        config_files = listdir(CONFIG_DIR)
        
        if MODEL_FILE_NAME not in config_files: raise MissingModelFile('Expected ' + MODEL_FILE_NAME + ' in ' + CONFIG_DIR)
        if CONFIG_FILE_NAME not in config_files: raise MissingConfigFile('Expected ' + CONFIG_FILE_NAME + ' in ' + CONFIG_DIR)
    
    if img_dataset_flag or split_img_dataset_flag:
   
        if img_dataset_flag:
            
            if not isdir(IMG_DATASET): 
            
                raise IOError('Image dataset directory does not exist !')
        
            if PERCENT is None and img_dataset_flag:
            
                raise ValueError('--' + PERCENT_ARG + ' is required')
            
            if not (TRAIN_DS_PERCENT_MIN <= PERCENT <= TRAIN_DS_PERCENT_MAX):
            
                raise ValueError('--' + PERCENT_ARG + ' value should be between ' + str(TRAIN_DS_PERCENT_MIN) + ' and ' + str(TRAIN_DS_PERCENT_MAX)) 
        
        
        if split_img_dataset_flag:
           
            missing_tr = not isdir(TRAIN_IMG_DATASET)
            missing_ev = not isdir(EVAL_IMG_DATASET)
            
            common = 'dataset directory does not exist !'
            if missing_tr or missing_ev:
             
                if missing_tr and missing_ev: msg = 'Train and Evaluation datasets directories do not exist !'
                
                if missing_tr: msg = 'Train ' + common
                    
                if missing_ev: msg = 'Eval ' + common
              
                raise IOError(msg)
        
        
        if THRESHOLD is None:
            
            raise ValueError('--' + THRESHOLD_ARG + ' is required')
        
    
        if not(MEM_THRESHOLD_PERCENT_MIN <= THRESHOLD <= MEM_THRESHOLD_PERCENT_MAX):
            
            raise ValueError('--' + THRESHOLD_ARG + ' value should be between ' + str(MEM_THRESHOLD_PERCENT_MIN) + ' and ' + str(MEM_THRESHOLD_PERCENT_MAX))
    
    if tfr_dataset_flag: # if we reached so far with valid combos, the following have to hold
        
        if not isdir(TFR_DATASET): 
            
            err_msg = 'TFR dataset directory does not exist !'
            
            raise IOError(err_msg)
        
        tfr_dataset_dir_contents = listdir(TFR_DATASET)
    
        if METADATA_FILE_NAME not in tfr_dataset_dir_contents : raise MissingMetaDataFile('Expected name should be ' + METADATA_FILE_NAME)
        
        with open(join(TFR_DATASET, METADATA_FILE_NAME), 'r') as handler:
        
            metadata = load(handler)

        annotation_ds_flag = metadata['ANNOTATION_DS_FLAG']
    
        if TASK == DTT and not annotation_ds_flag             : raise ModelDatasetMissmatch('Cannot train a yolo detector on a regular image dataset ment for a classifier')
        
        potential_train_tfr_shards = glob(join(TFR_DATASET, '*' + TRAIN_TFR_NAME))
        if not potential_train_tfr_shards: raise MissingTrainTFR('Expected at least one train tfr shard with the substring ' + TRAIN_TFR_NAME + ' as a substring in the file name')

        potential_eval_tfr_shards = glob(join(TFR_DATASET, '*' + EVAL_TFR_NAME))
        if not potential_eval_tfr_shards: raise MissingTrainTFR('Expected at least one eval tfr shard with the substring ' + EVAL_TFR_NAME + ' as a substring in the file name')


    if TFR_DEST_DIR is None:
        TFR_DEST_DIR = DUMP_DIR
        
        
    print('Done !')
    
    return (
        
        MODE, TASK, CONFIG_DIR, DUMP_DIR, TFR_DEST_DIR, 
        IMG_DATASET, PERCENT, TRAIN_IMG_DATASET, EVAL_IMG_DATASET, 
        THRESHOLD, TFR_DATASET, SRC_DIR, DST_DIR, EXPORT_MODEL_DIR, 
        EXPORT_MODEL_FILE_NAME, HIDE_META, TOCO_COMPATIBLE, TFR_NUM_SHARDS
    
    )

def check_config_validity(config_file):
    
    # TODO
    pass

def update_config(config_file, entry, key=None):
    
    '''
    
        Update or append specific entries in config.json
    
    '''
    
    with open(config_file, 'r') as config_json:
        
        config_data = load(config_json)

    if key is None: config_data.update(entry)
    else: config_data[ key ].update(entry) 

    with open(config_file, 'w') as config_json:
    
        dump(config_data, config_json, indent=5, sort_keys = True)
        

def yolo_loss(y_pred, y_true, true_boxes, yolo_spec):

    # might have to make the computations in float64 since in some scenarios where the model is weak, 
    # the loss for the regression part will blow up to Nan

    # tmp = tf.Print(y_true, [tf.shape(y_true), tf.shape(true_boxes), tf.shape(y_pred)], summarize = 50)

    num_anchors = int(len(yolo_spec['ANCHORS']) / 2)

    mask_shape = tf.shape(y_true)[:4]
    batch_size = tf.shape(y_pred)[0]  # since it can differ, better compute it at runtime, for instance if num_examples % batch_size != 0 we are screwed, might have to make batching to be based on a NUM_BATCHES system relative to the dataset

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(yolo_spec['GRID_W']), [ yolo_spec['GRID_H'] ]), (1, yolo_spec['GRID_H'], yolo_spec['GRID_W'], 1, 1)))
    
    cell_y = tf.to_float(tf.reshape(tf.tile(tf.reshape(tf.range(yolo_spec['GRID_H']), (yolo_spec['GRID_H'], 1)), [ 1, yolo_spec['GRID_W'] ]), (1, yolo_spec['GRID_H'], yolo_spec['GRID_W'], 1, 1))) 

    cell_grid = tf.tile(tf.concat([ cell_x, cell_y ], -1), [ batch_size, 1, 1, num_anchors, 1 ])

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    """
    Adjust prediction
    """
    # ## adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    # ## adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(yolo_spec['ANCHORS'], [1, 1, 1, num_anchors, 2])

    # ## adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    # ## adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    # ## adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

    # ## adjust w and h
    true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

    # ## adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    # ## adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    # ## coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * yolo_spec['COORD_SCALE']

    # ## confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * yolo_spec['NO_OBJECT_SCALE']

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * yolo_spec['OBJECT_SCALE']

    # ## class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(yolo_spec['CLASS_WEIGHTS'], true_box_class) * yolo_spec['CLASS_SCALE']

    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < yolo_spec['COORD_SCALE'] / 2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, yolo_spec['WARM_UP_BATCHES']),
                                                   lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                            true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                yolo_spec['ANCHORS'], [1, 1, 1, num_anchors, 2]) * no_boxes_mask,
                                                            tf.ones_like(coord_mask)],
                                                   lambda: [true_box_xy,
                                                            true_box_wh,
                                                            coord_mask])

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    # the add op version of the loss yields ' Failed to run optimizer ArithmeticOptimizer '
    #loss = loss_xy + loss_wh + loss_conf + loss_class
    
    # this one doesn't but i can't vouch for wether it's working as intended or not yet !!! Can't see why it would not tho :D
    loss = tf.reduce_sum([ loss_xy, loss_wh, loss_conf, loss_class ])

    #nb_true_box = tf.reduce_sum(y_true[..., 4])
    #nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """
    
    # current_recall = nb_pred_box / (nb_true_box + 1e-6)
    # total_recall = tf.assign_add(total_recall, current_recall)
    #
    #loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    #loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    #loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    #loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    #loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    #loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    #loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    #loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss


def append_yolo_layer(current_output, num_anchors, num_classes, grid_w, grid_h):

    # might have different implicit values for the parameters compared to the keras implementation of Conv2D, have to do some checking !!!
    
    yolo_layer_output = tf.layers.conv2d(
        
                        inputs = current_output,
                        filters = num_anchors * (4 + 1 + num_classes),
                        kernel_size = 1,
                        padding = 'same',
                        strides = 1,
                        activation = None,
                        kernel_regularizer = None,
                        bias_regularizer = None,
                        name = 'yolo_layer'
                        
    )
    
    return tf.reshape(yolo_layer_output, [ -1, grid_h, grid_w, num_anchors, 4 + 1 + num_classes ], name = OUTPUT_OP_NAME)

def append_logits_layer(current_output, num_classes, train_flag):

    reg = tf.contrib.layers.l2_regularizer( scale = 0.00005 )
    return tf.layers.dense(inputs = current_output, units = num_classes, kernel_regularizer = reg, bias_regularizer = reg, activation = None, trainable = train_flag, name = 'logits')

def build_estimator_model_function(features, labels, mode, params):

    '''
    
        TODO: 
        
            Find the best way to fetch learning rate for without imposing any constraints on model.py definition.
            
            Worst case scenario, a simple standard name for the learning rate variable / constant should do the trick,
            since one can just fetch it from the graph in order to log it.
        
    '''

    YOLO_MODEL = params['TASK'] == DTT
    
    user_defined_model_functon = params['MODEL_FUNCTION']

    current_model_output, user_defined_optimizer_op = user_defined_model_functon(
                                                                                 
                features,
                train_flag = ( mode == tf.estimator.ModeKeys.TRAIN )
                                                            
    )
    
    reg_loss_name = 'regularization_loss'
    reg_loss_collection = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    reg_loss = tf.constant(0, name = reg_loss_name, dtype = tf.float32)

    if reg_loss_collection: 
        reg_loss = tf.add_n(reg_loss_collection, name = reg_loss_name)
    
    if YOLO_MODEL:
 
        prediction = append_yolo_layer(
            
            current_model_output,
            int (len(params['YOLO_SPEC']['ANCHORS']) / 2),
            params['TRAIN_INFO']['NUM_CLASSES'],
            params['YOLO_SPEC']['GRID_W'],
            params['YOLO_SPEC']['GRID_H']
        )


        '''
            
            labels[0] -> y_batch , grid relative coords for bboxes, bundled up with respect to the grid cells
            labels[1] -> b_batch , grid relative coords for bboxes, bundled up with no with no consideration for the grid cells
            labels[2] -> gt_batch , absolute [ xmin, ymin, xmax, ymax, ... ] values for bboxes
            
        '''

        loss = yolo_loss(prediction, labels[0], labels[1], params['YOLO_SPEC'])
    
    else: # I.E. CLS

        logits = append_logits_layer(

            current_model_output, 
            params['TRAIN_INFO']['NUM_CLASSES'], 
            train_flag = mode == tf.estimator.ModeKeys.TRAIN

        )
        
        if mode != tf.estimator.ModeKeys.PREDICT:

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)
        
        prediction = tf.nn.softmax(logits=logits, name=OUTPUT_OP_NAME)
        
    if mode != tf.estimator.ModeKeys.PREDICT:

        total_loss = tf.add(loss, reg_loss, name = 'total_loss')

    if params['QUANTIZE_DELAY'] > 0:
        
        if mode == tf.estimator.ModeKeys.TRAIN: tf.contrib.quantize.create_training_graph( quant_delay = params['QUANTIZE_DELAY'])
        else: tf.contrib.quantize.create_eval_graph()

    if mode == tf.estimator.ModeKeys.PREDICT:
    
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)
        
    else :
        
        if YOLO_MODEL:

            ( 
             
              iou_tensor,        
              iou_update_op,     
              batch_true_pos_tensor, 
              batch_false_pos_tensor, 
              batch_false_neg_tensor 
              
              
              ) = get_detector_metrics(
                
                    prediction, 
                    labels[2],
                    params['TRAIN_INFO']['NUM_CLASSES'], 
                    params['YOLO_SPEC'],
                    params['MODEL_INPUT_WIDTH'],
                    params['MODEL_INPUT_HEIGHT']
                
            )
            
            sess_run_hook = DetectorSessionRunHook( 
            
                    iou_tensor, 
                    iou_update_op,
                    batch_true_pos_tensor, 
                    batch_false_pos_tensor, 
                    batch_false_neg_tensor,
                    loss,
                    reg_loss,
                    total_loss,
                    tf.train.get_global_step(),
                    params['CKPT_DIR'],
                    params['MAX_TOLERANCE_COUNT'],
                    mode == tf.estimator.ModeKeys.TRAIN,
                    params['HIDE_META']
            
            )
              
        else:
        
            (
             
             acc_tensor,
             acc_update_op
             
             ) = tf.metrics.accuracy(
                                    
                    labels = tf.argmax(labels, 1), 
                    predictions = tf.argmax(prediction, 1), 
                    name = 'acc_metric_op'
                    
            )
            
            sess_run_hook = ClassifierSessionRunHook( 
            
                    acc_tensor, 
                    acc_update_op,
                    loss,
                    reg_loss,
                    total_loss,
                    tf.train.get_global_step(),
                    params['CKPT_DIR'],
                    params['MAX_TOLERANCE_COUNT'],
                    mode == tf.estimator.ModeKeys.TRAIN,
                    params['HIDE_META']
            
            )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(update_ops):
            
                optimizer_op = user_defined_optimizer_op.minimize( loss = total_loss, global_step = tf.train.get_global_step() )

        else:

            optimizer_op = user_defined_optimizer_op.minimize( loss = total_loss, global_step = tf.train.get_global_step() )
        
        
        spec = tf.estimator.EstimatorSpec(
                                          
                mode = mode, 
                loss = total_loss, 
                train_op = optimizer_op,
                training_hooks = [sess_run_hook],
                evaluation_hooks = [sess_run_hook]
        )
        
    return spec

