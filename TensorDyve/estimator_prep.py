import numpy as np
import tensorflow as tf

from os import listdir, mkdir
from os.path import dirname, join, basename, isfile, isdir, splitext
from random import randint, uniform

from cv2 import imread, IMREAD_COLOR, IMREAD_GRAYSCALE, imwrite, resize, \
    rectangle, putText, FONT_HERSHEY_SIMPLEX, copyMakeBorder, BORDER_CONSTANT, INTER_LINEAR

from imgaug import Keypoint, KeypointsOnImage

from tqdm import tqdm
from glob import glob
from json import load

from .project_globals import TFR_SHAPE_KEY, TFR_SAMPLE_KEY, \
    TFR_GROUND_TRUTH_KEY, BUTS_DIR_NAME, TFR_IMPLICIT_DIR_NAME, JSON_ANNOTATION_DIR, ALLOWED_EXTENSIONS

from .utils import bbox_iou, crop_image, sigmoid, softmax, convert_grid_relative_bboxes_from_netout, get_absolute_bbox
from .dataset_utils import AnnotParser

class EstimatorInputPipe():
    
    def __init__(self, cls_flag, annot_flag, aug_pipe, batching_params, num_classes, yolo_spec, model_input_image_shape):
        
        self._aug_pipe = aug_pipe
        self._batching_params = batching_params
        self._num_classes = num_classes
        
        self._model_input_image_shape = model_input_image_shape
        
        self._model_input_width = model_input_image_shape[1]
        self._model_input_height = model_input_image_shape[0] 
        self._model_input_channels = model_input_image_shape[2]
        
        self._cls_flag = cls_flag
        self._annot_flag = annot_flag
        
        self.yolo_spec = yolo_spec
        
        self._ground_truth_dtype = tf.float32
        self._image_dtype = tf.uint8
        self._shape_dtype = tf.int32
        
        self._features = { 
            
            TFR_SHAPE_KEY          : tf.FixedLenFeature([], tf.string), 
            TFR_SAMPLE_KEY         : tf.FixedLenFeature([], tf.string),
            TFR_GROUND_TRUTH_KEY  : tf.FixedLenFeature([], tf.string) if annot_flag else tf.FixedLenFeature([], tf.int64) 
                
        }

    def _decode_image_shape(self, raw_image_shape):
        
        return tf.decode_raw(raw_image_shape, self._shape_dtype)

    def _build_image(self, raw_image, image_shape):
        
        image = tf.decode_raw(raw_image, self._image_dtype)
      
        return tf.reshape(image, image_shape) # reshaping flattened version of the image
    
    def _build_ground_truth(self, raw_ground_truth):
        
        if self._annot_flag:
            
            '''
        
            annotations for one image --> [ [xmin, xmax, ymin, ymax, cls_index], [xmin, xmax, ymin, ymax, cls_index], [xmin, xmax, ymin, ymax, cls_index], ... ]
            
            * after decode we get the flattened array, of which the last element is the number of annotations
            * poping last element and using it to reshape the remaining array to [num_annotations, elements_per_annotation]
                            
            '''
        
            annotations = tf.decode_raw(raw_ground_truth, self._ground_truth_dtype)
            num_elements = tf.size(annotations)
            num_annotations = annotations[ num_elements - 1 ]
            ground_truth = tf.slice(annotations, [0], [ num_elements - 1 ])

            ground_truth = tf.reshape(ground_truth, [ num_annotations, tf.cast(6, tf.int32) ] )
            
        else :
        
            ground_truth = tf.cast(raw_ground_truth, tf.int64)
            
            ground_truth = tf.one_hot(
            
                    indices = ground_truth,
                    depth = self._num_classes,
                    on_value = tf.cast(1, self._ground_truth_dtype),
                    off_value = tf.cast(0, self._ground_truth_dtype),
                    dtype = self._ground_truth_dtype
                    
            )
        
        return ground_truth
    
    def _deserialize(self, serialized):

        parsed_example = tf.parse_single_example(serialized, self._features)
        
        image_shape   = self._decode_image_shape( parsed_example[ TFR_SHAPE_KEY ] )
        image         = self._build_image( parsed_example[ TFR_SAMPLE_KEY ], image_shape)
        ground_truth = self._build_ground_truth(parsed_example[ TFR_GROUND_TRUTH_KEY ])
        
        # this hack returns a tensor that HAS a shape attribute, otherwise the resize method used after will complain about not having one
        # this is not going to do any resize per say, since the shape mentioned is the actual image shape
        
        image = tf.image.resize_image_with_crop_or_pad(image, image_shape[1], image_shape[0])
       
        return image, ground_truth


    def _yolo_batching_wrapper(self):

        def _yolo_batching(image, ground_truths):
        
            # at this point image has been resized to model input size
            
            im_width = image.shape[1]
            im_height = image.shape[0]
        
            true_box_index = 0
            
            b_label = np.zeros((1, 1, 1, self.yolo_spec['TRUE_BOX_BUFFER'], 4), dtype=np.float32)  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
            y_label = np.zeros((self.yolo_spec['GRID_H'], self.yolo_spec['GRID_W'], int(len(self.yolo_spec['ANCHORS']) / 2), 4 + 1 + self._num_classes), dtype=np.float32)  # desired network output
            
            if np.sum(ground_truths[0]) == 0:# I.E. posible negative example [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ]
                
                return image.astype( np.float32 ) / 255. ,y_label, b_label, np.asarray( ground_truths, dtype = np.float32 )
            
            else:
                
                # xmin, ymin, xmax, ymax is the original implementation using BoundBox object 
                anchors = [ [0, 0, self.yolo_spec['ANCHORS'][2 * i], self.yolo_spec['ANCHORS'][ 2 * i + 1 ] ] for i in range( len( self.yolo_spec['ANCHORS'] ) // 2 ) ]
                
                ground_truths = ground_truths.astype(np.int32)  # if ground_truths dtype == uint8 --> overflow on arithmetic ops like ( xmin + xmax ) * 0.5 wtfyayo
                
                # if aug flag is present, box coords have to be converted to absolute values in order to adjust by keypoints after augs, else they are normalzied coords and have to be converted given new image shape after resize
                scale_w = 1 if self._annot_flag else im_width
                scale_h = 1 if self._annot_flag else im_height
                
                for ground_truth in ground_truths:
                    
                    xmin = ground_truth[0] * scale_w
                    ymin = ground_truth[1] * scale_h
                    xmax = ground_truth[2] * scale_w
                    ymax = ground_truth[3] * scale_h
                    
                    cls_index = ground_truth[5] # obj_prob is 4 th, filled that in just for consistency since a bbox representing the prediction has that in there also
                    
                    if xmax > xmin and ymax > ymin:
                        
                        center_x = .5 * (xmin + xmax)
                        center_x = center_x / ( im_width / self.yolo_spec['GRID_W'] )
                        
                        center_y = .5 * (ymin + ymax)
                        center_y = center_y / ( im_height / self.yolo_spec['GRID_H'] )
                
                        grid_x = int( np.floor( center_x ) )
                        grid_y = int( np.floor( center_y ) )
                
                        if grid_x < self.yolo_spec['GRID_W'] and grid_y < self.yolo_spec['GRID_H']:
                            
                            center_w = (xmax - xmin) / (float(im_width) / self.yolo_spec['GRID_W'])  # unit: grid cell
                            center_h = (ymax - ymin) / (float(im_height) / self.yolo_spec['GRID_H'])  # unit: grid cell
                            
                            bbox = [ center_x, center_y, center_w, center_h ]
                
                            # find the anchor that best predicts this bbox
                            best_anchor = -1
                            max_iou = -1
                
                            shifted_box = [0, 0, center_w, center_h ]
                            
                            for i in range(len(anchors)):
                                
                                anchor = anchors[i]
                                iou = bbox_iou(shifted_box, anchor)
                                
                                if max_iou < iou:
                                    
                                    best_anchor = i
                                    max_iou = iou
                                    
                            # assign ground truth x, y, w, h, confidence and class probs to y_label
                            y_label[ grid_y, grid_x, best_anchor, 0:4 ] = bbox
                            y_label[ grid_y, grid_x, best_anchor, 4  ] = 1.
                            y_label[ grid_y, grid_x, best_anchor, 5 + cls_index ] = 1
                            
                            # assign the true bbox to b_label
                            b_label[0, 0, 0, true_box_index] = bbox
                            
                            true_box_index += 1
                            true_box_index = true_box_index % self.yolo_spec['TRUE_BOX_BUFFER']
                             
                             
                return image.astype( np.float32 ) / 255., y_label, b_label, np.asarray(ground_truths, dtype = np.float32)
    
        return _yolo_batching

    def _augment_batch_with_labels_wrapper(self):
        
        # tf py_func cant convert py objects like _aug_pipe to tensor, that's why we need this wrapper so that the object is in scope and ready to be used in the actual function, which is defined inside the wrapper func
        
        def _augment_batch(images, labels):
                
            return self._aug_pipe.augment_image(images), labels
    
        return _augment_batch
    
    def _augment_batch_with_annot_wrapper(self):
        
        def _augment_batch(images, ground_truths):
                      
            images = self._aug_pipe.augment_image(images)
            
            image_shape = images.shape
            
            keypoints_on_images = []
            keypoints = []
             
            # for each image, ground_truths = [ ann1 -> [xmin, ymin, xmax, ymax, cls_index], ...]
            
            im_w = image_shape[1]
            im_h = image_shape[0]
            
            for label in ground_truths:
                
                if np.sum(label) == 0: # possible negative example [ [0, 0, 0, 0, 0, 0] ]
                    
                    continue
                
                _label = [ label[0] * im_w, label[1] * im_h, label[2] * im_w, label[3] * im_h ]
                
                keypoints.append(Keypoint(x=_label[0], y=_label[1]))  # top left xmin, ymin
                keypoints.append(Keypoint(x=_label[2], y=_label[3]))  # bottom right xmax, ymax
                keypoints.append(Keypoint(x=_label[0], y=_label[3]))  # bottom left xmin, ymax
                keypoints.append(Keypoint(x=_label[2], y=_label[1]))  # top right xmax, ymin
        
            keypoints_on_images.append(KeypointsOnImage(keypoints, shape=image_shape))
            keypoints_on_images = det_aug.augment_keypoints(keypoints_on_images)
        
            index = 0
            for keypoint in keypoints_on_images[0].keypoints:
                 
                if index % 4 == 0:
                    
                    x1, y1 = keypoint.x, keypoint.y
                     
                if index % 4 == 1:
                    
                    x2, y2 = keypoint.x, keypoint.y
         
                if index % 4 == 2:
                    
                    x3, y3 = keypoint.x, keypoint.y
                    
                if index % 4 == 3:
                    
                    x4, y4 = keypoint.x, keypoint.y
                    
                    xmin = min(x1, x2, x3, x4)
                    xmax = max(x1, x2, x3, x4)
                    ymin = min(y1, y2, y3, y4)
                    ymax = max(y1, y2, y3, y4)
        
                    ground_truths[ int(index / 4) ][0] = xmin
                    ground_truths[ int(index / 4) ][1] = ymin
                    ground_truths[ int(index / 4) ][2] = xmax
                    ground_truths[ int(index / 4) ][3] = ymax
                     
                index += 1
                
            return images, ground_truths
    
        return _augment_batch
    
    
    def _build_yolo_batching_py_func(self, image, ground_truth):
    
        py_func_op = tf.py_func(func=self._yolo_batching_wrapper(), inp=[ image, ground_truth ], Tout=(np.float32, np.float32, np.float32, np.float32))
        py_func_op[0].set_shape(image.shape)
        
        return py_func_op
    
    def _build_aug_py_func(self, image, ground_truth):

        aug_func = self._augment_batch_with_annot_wrapper() if self._annot_flag else self._augment_batch_with_labels_wrapper()

        py_func_op = tf.py_func(func=aug_func, inp=[ image, ground_truth ], Tout=(np.uint8, np.float32))
        
        py_func_op[0].set_shape(image.get_shape())
        
        return py_func_op

    def _build_crop_image_py_func(self, image, annotations):

        # relocate crop image to utils, can use it in predict for classifier also
        
        resize_shape = ( self._model_input_image_shape[1], self._model_input_image_shape[0] )
        py_func_op = tf.py_func( func = crop_image, inp = [ image, annotations, resize_shape ], Tout = (np.uint8, np.int32))
        
        py_func_op[0].set_shape( resize_shape ) # op output a crop of size 'resize_shape'
        
        return py_func_op
    
   
    def _build_classifier_pipe(self, dataset, train_flag):
        
        dataset = dataset.filter(lambda img, gt: tf.greater( tf.reduce_sum(gt), 0) ) # filter out negative examples with gt [ [0, 0, 0, 0, 0, 0] ]
        
        if train_flag:
            
            if self._aug_pipe is not None:
                
                dataset = dataset.map(lambda img, gt: self._build_aug_py_func(img, gt), self._batching_params['NR_CPU_CORES'])
                
            dataset = dataset.shuffle(buffer_size=self._batching_params['SHUFFLE_BUFFER_SIZE'])
            dataset = dataset.prefetch(self._batching_params['PREFETCH_POOL_SIZE'])
            num_repeat = None
        
        else :
            
            num_repeat = 1
    
        if self._annot_flag:
                
                # batch up annotated images, so that the crop_image function can get ALL the crops from them and build the actual batch that will be fed into the training loop
                dataset = dataset.map(lambda img, gt: self._build_crop_image_py_func(img, gt), self._batching_params['NR_CPU_CORES'])
                dataset = dataset.map(
                                      
                            lambda img, gt: ( img, tf.one_hot(
            
                                                    indices   = gt,
                                                    depth     = self._num_classes,
                                                    on_value  = tf.cast(1, self._ground_truth_dtype),
                                                    off_value = tf.cast(0, self._ground_truth_dtype),
                                                    dtype     = self._ground_truth_dtype
                                                )
                            ),

                            self._batching_params['NR_CPU_CORES'])
            
        dataset = dataset.map( 
            
            lambda img, gt: ( tf.image.resize_images(img, (self._model_input_height, self._model_input_width)), gt), 
            self._batching_params['NR_CPU_CORES']
            
        )

        dataset = dataset.map(lambda img, gt: (tf.divide(tf.cast(img, tf.float32), 255.), gt), 1)
        
        dataset = dataset.prefetch(self._batching_params['PREFETCH_POOL_SIZE'])
        
        dataset = dataset.repeat(num_repeat)
        
        dataset = dataset.batch( batch_size = self._batching_params['BATCH_SIZE'] if train_flag else 1 )
        
        dataset = dataset.prefetch( self._batching_params['PREFETCH_POOL_SIZE'] )
        
        dataset = dataset.apply( # aparetly this is not just another joke feature, this is really gud :D
            
            tf.data.experimental.prefetch_to_device(
                device='/gpu:0',
                buffer_size=None
            )
        )
        
        iterator = dataset.make_one_shot_iterator()
        
        image, label = iterator.get_next()
        
        # another redundant call in order to explicitly have a static shape for the whole thing to work
        image = tf.reshape(image, [-1] + self._model_input_image_shape ) # todo: replace with tf.data assert shape
        
        return image, label

    def _build_detector_pipe(self, dataset, train_flag):
        
        dataset = dataset.map( 
            
            lambda img, gt: ( tf.image.resize_images(img, (self._model_input_height, self._model_input_width)), gt), 
            self._batching_params['NR_CPU_CORES'] 
            
        )
        
        if train_flag:
            
            if self._aug_pipe is not None:
                
                dataset = dataset.map( lambda img, gt: self._build_aug_py_func( img, gt ), self._batching_params['NR_CPU_CORES'] )
            
            dataset = dataset.shuffle ( buffer_size = self._batching_params['SHUFFLE_BUFFER_SIZE'] )
            dataset = dataset.prefetch( self._batching_params['PREFETCH_POOL_SIZE'] )
            num_repeat = None
        
        else :
            
            num_repeat = 1
    
        dataset = dataset.map     ( lambda img, gt: self._build_yolo_batching_py_func( img, gt ), self._batching_params['NR_CPU_CORES'] )
        dataset = dataset.repeat  ( num_repeat )
        dataset = dataset.batch   ( batch_size = self._batching_params['BATCH_SIZE'] )
        dataset = dataset.prefetch( self._batching_params['PREFETCH_POOL_SIZE'] )
        
        iterator = dataset.make_one_shot_iterator()
        
        image, y_gt, b_gt, gt = iterator.get_next()
        
        # another redundant call in order to explicitly have a static shape for the whole thing to work
        image = tf.reshape(image, [-1] + self._model_input_image_shape)
        
        return image, ( y_gt, b_gt, gt )


    def build_tfr_pipe(self, tfr_regex_path, train_flag):
        
        '''
    
            invoke when catching the MemThreshold exception
    
        '''
        
        '''
        
        tfr_files_dataset = tf.data.Dataset.list_files(tfr_regex_path, shuffle = train_flag)

        dataset = tfr_files_dataset.apply(
            
            tf.data.experimental.parallel_interleave(
                
                map_func = lambda tfr_file: tf.data.TFRecordDataset(tfr_file), 
                cycle_length = self._batching_params['NR_CPU_CORES'],
                block_length = 1 # this is useful if batch class uniform distribution is to be achieved with one tfr per class, maybe the class tfr even being sharded
                                                
            )
            
        )

        '''

        dataset = tf.data.TFRecordDataset(glob(tfr_regex_path))

        #options = tf.data.Options()
        #options.experimental_threading.private_threadpool_size = 20
        #dataset = dataset.with_options(options) # have not seen any improvements, maybe rethinking the inputpipelie somehow might yield some benefits due to threading


        dataset = dataset.prefetch( self._batching_params['PREFETCH_POOL_SIZE'] * 2)

        dataset = dataset.map( lambda serialized: self._deserialize( serialized ), self._batching_params['NR_CPU_CORES'] )

        dataset = dataset.prefetch( self._batching_params['PREFETCH_POOL_SIZE'] * 2) 

        return self._build_classifier_pipe(dataset, train_flag) if self._cls_flag else self._build_detector_pipe(dataset, train_flag)

    def build_mem_pipe(self, images, ground_truths, train_flag):
        
        '''
        
            invoke after loading dataset in memory procedure was succesfull 
        
        '''
        
        return self._build_classifier_pipe(tf.data.Dataset.from_tensor_slices((images, ground_truths)), train_flag) if self._cls_flag else self._build_detector_pipe(tf.data.Dataset.from_tensor_slices((images, ground_truths)), train_flag)

class ClassifierEstimator(tf.estimator.Estimator):

    def _dir_check(self, dir):
        
        # maybe bundle up the dir path construction also, using os.path.join, removing some of the duplicate code in predict_wrapper
        if not isdir(dir): mkdir(dir)
    
    def _get_generatorX(self, image_files, width, height, channels):
        
        imread_mode = IMREAD_COLOR if channels == 3 else IMREAD_GRAYSCALE
        
        def _build_generatorX():
        
            for image_file in tqdm(image_files):
                
                image = imread(image_file, imread_mode)
                image = resize(image, (width, height))
                normalized_image = image.astype(np.float32) / 255.
                
                if channels == 1:
                    
                    normalized_image = np.expand_dims(normalized_image, 2)  # account for 2 dim GRAYSCALE read, (width, height) only
                
                yield np.expand_dims(normalized_image, 0) # batch it up, the input for the estimator model must be 4 dim, I.E. batch dim included
                

        return _build_generatorX
                
    def _build_predict_input_fnX(self, image_files, width, height, channels):

        # return a callable for the estimator api to use
        return lambda: tf.data.Dataset.from_generator(self._get_generatorX(image_files, width, height, channels), output_types = (tf.float32), output_shapes = ( tf.TensorShape( [None, width, height, channels] )) )
     
    def predict_wrapper(self, src_dir, dst_dir):
    
        image_files = []  # string list, paths to image files
        labels = []  # integer list, class index in cls prob distribution 
        
        imread_mode = IMREAD_COLOR if self.params['MODEL_INPUT_CHANNELS'] == 3 else IMREAD_GRAYSCALE
        
        class_label_dirs = self.params['TRAIN_INFO']['CLASS_NAMES']

        num_class_labels = self.params['TRAIN_INFO']['NUM_CLASSES']
        
        num_removed = 0
        
        # remove potential hidden files and/or folders, or the implicit TFR_DATASET folder that might be present due to not being able to load dataset into memory
        for i in range(num_class_labels):
        
            actual_index = i - num_removed
            class_label_dir = class_label_dirs[ actual_index ]
        
            if class_label_dir.startswith('.') or (class_label_dir == TFR_IMPLICIT_DIR_NAME):
        
                num_removed += 1
                class_label_dirs.pop(actual_index)

        if not class_label_dirs:
            
            for ext in ALLOWED_EXTENSIONS:
                
                image_files += glob(join(src_dir, '*' + ext))
            
        else:
            
            #class_label_dirs.sort()
            
            num_classes = len(class_label_dirs)
            
            for i in range(num_classes):
                
                ith_class_image_files = []
                
                for ext in ALLOWED_EXTENSIONS:
                
                    ith_class_image_files += glob(join(join(src_dir, class_label_dirs[i]), '*' + ext))
                
                nr_samples = len(ith_class_image_files)

                ith_class_labels = [i] * nr_samples

                image_files += ith_class_image_files
                labels += ith_class_labels
    
        num_images = len(image_files)

        prediction_generator = self.predict(

            self._build_predict_input_fnX(

                image_files,
                self.params['MODEL_INPUT_HEIGHT'], 
                self.params['MODEL_INPUT_WIDTH'], 
                self.params['MODEL_INPUT_CHANNELS']

            ),

            yield_single_examples = True

        )
        
        net_outputs = []
        num_examples = 0
        
        try:
            
            print('Computing predictions for ', num_images, 'images ...')
            
            while True:
                
                    net_outputs.append(next(prediction_generator))
                    num_examples += 1
                
        except StopIteration:
                
            print ('Finished !')
            
        print('Writing images to destination directory based on the predicted class ...')
            
        predictions = []

        for i in tqdm(range(num_examples)):
            
            image_file = image_files[i]
            net_out = net_outputs[i]

            predicted_class_index = net_out.argmax(axis=0)

            image = imread(image_file, imread_mode)
            
            class_label = self.params['TRAIN_INFO']['CLASS_NAMES'][predicted_class_index]
            image_file_name = basename(image_file)

            if labels:
                
                match = predicted_class_index == labels[i]
                predictions.append(match)
                
                if match:
                    
                    pred_dst_dir = join(dst_dir, class_label)
                    
                    self._dir_check(pred_dst_dir)
                    
                    pred_dst_file = join(pred_dst_dir, image_file_name)

                    if not imwrite(pred_dst_file, image):
                        
                        print('Failed to write image to ', pred_dst_file)
                        
                else:
                    
                    buts_dir = join(dst_dir, BUTS_DIR_NAME)
                    self._dir_check(buts_dir)
                    
                    but_dir_name = self.params['TRAIN_INFO']['CLASS_NAMES'][predicted_class_index] + ' BUT ' + self.params['TRAIN_INFO']['CLASS_NAMES'][ labels[i] ]
                    
                    specific_but_dir = join(buts_dir, but_dir_name)
                    self._dir_check(specific_but_dir)
                
                    specific_but_file = join(specific_but_dir, image_file_name)
                
                    if not imwrite(specific_but_file, image):
                        
                        print('Failed to write image to ', specific_but_file)
                        
            else:
                
                pred_dst_dir = join(dst_dir, class_label)
                self._dir_check(pred_dst_dir)
                
                pred_dst_file = join(pred_dst_dir, image_file_name)
                
                if not imwrite(pred_dst_file, image):
                    
                    print('Failed to write image to ', pred_dst_file)

        predictions = np.asarray(predictions, dtype = np.float32)
        
        accuracy = np.mean(predictions, axis = 0)
                
        with open( join( dst_dir, 'ACCURACY: ' + str(accuracy) ), 'w') as dummy_handler:
                    
            # just wanted to create an empty file that will have a name that will encode the accuracy metric for, easy lookup, in dst_dir
            pass

class YoloEstimator(ClassifierEstimator):
    
    def _draw_boxes(self, image, bboxes, color = (0, 255, 0), y_shift = 10):

        image_h, _, _ = image.shape

        for bbox in bboxes :
            
            if np.sum(bbox) == 0: # skip possible negative example [[0, 0, 0, 0, 0, 0]]
                
                continue
            
            xmin, ymin, xmax, ymax = bbox[:4]
    
            rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            putText(
                
                image,
                str( np.argmax( bbox[5:] ) ) + ' ' + str( bbox[4] ),
                (xmin, ymin - y_shift),
                FONT_HERSHEY_SIMPLEX,
                1.4e-3 * image_h,
                color, 
                1
            )
            
        return image

    def _decode_netout(self, net_out, num_classes, yolo_spec):
        
        boxes = convert_grid_relative_bboxes_from_netout(net_out, num_classes, yolo_spec)
        
        # N M S
        for c in range(num_classes):
            
            sorted_indices = list(reversed(np.argsort([ bbox[ ..., 5: ][ c ] for bbox in boxes ]))) 
    
            num_sorted_indices = len(sorted_indices)
    
            for i in range(num_sorted_indices):
                
                index_i = sorted_indices[ i ]
                
                if boxes[ index_i ][..., 5:][c] == 0 or boxes[ index_i ][ 4 ] <= yolo_spec['OBJ_THRESHOLD']:
                     
                    continue
                
                else:
                    
                    for j in range(i + 1, num_sorted_indices):
                        
                        index_j = sorted_indices[ j ]
                        
                        if  bbox_iou(boxes[ index_i ], boxes[ index_j ]) >= yolo_spec['NMS_THRESHOLD']:
                            
                            if np.argmax(boxes[ index_j ][..., 5:]) == c:  
                                                                              
                                boxes[ index_j ][ 4 ] = 0
                                
                            boxes[index_j][..., 5:][ c ] = 0
        
        # Filtering out bboxes by obj probability                 
        boxes = [ bbox for bbox in boxes if bbox[ 4 ] > yolo_spec['OBJ_THRESHOLD'] ]
        
        return boxes

    def predict_wrapper(self, src_dir, dst_dir):
        
        image_files = []
        
        for ext in ALLOWED_EXTENSIONS:
        
            image_files += glob(join(src_dir, '*' + ext))
        
        annot_flag = False
        
        num_jsons = len( glob( join( join( src_dir, JSON_ANNOTATION_DIR ), '*_forTraining.json' ) ) )
        num_txts = len( glob( join( src_dir, '*.txt') ) )
        
        if num_jsons + num_txts > 0 :
            
            annot_flag = True
        
        num_images = len(image_files)
        
        print('Computing predictions for ', num_images, 'images ...')
        
        # TODO: make this to process batches of images and adapt the for each loop bellow, atm the predict is slow on gpu because it only proccesses one image at a time
        prediction_generator = self.predict(
                                            
                                        self._build_predict_input_fnX(
                                                                           
                                                            image_files,
                                                            self.params['MODEL_INPUT_HEIGHT'],
                                                            self.params['MODEL_INPUT_WIDTH'],
                                                            self.params['MODEL_INPUT_CHANNELS']),
                                             
                                        yield_single_examples = True  
                                        
                                    )
        
        net_outputs = []
        
        try:
            
            while True:
        
                net_outputs.append(next(prediction_generator))
        
        except StopIteration:
            
            print ('Finished !')
        
        print('Drawing bboxes on images ...')
        
        # TODO: x threads, examples / x to proccess by each thread
        
        for image_file, net_out in zip(tqdm(image_files), net_outputs):
            
            image = imread(image_file)
            
            ground_truths = [[0, 0, 0, 0, 0, 0]] # negative example
            
            bboxes = self._decode_netout(net_out, self.params['TRAIN_INFO']['NUM_CLASSES'], self.params['YOLO_SPEC'] )
            
            abs_bboxes = []
            
            for bbox in bboxes:
                
                abs_bboxes.append( get_absolute_bbox(bbox, self.params['MODEL_INPUT_WIDTH'], self.params['MODEL_INPUT_HEIGHT'] ) )
            
            annotated_image = self._draw_boxes(image, abs_bboxes)
            
            if annot_flag:

                image_file_name = basename(image_file)
                image_file_name = splitext(image_file_name)[0] # strip extension
                image_file_root_dir = dirname(image_file)

                json_annotation_file = join( join( image_file_root_dir, JSON_ANNOTATION_DIR ), image_file_name + '_forTraining.json' )
                txt_annotation_file  = join( image_file_root_dir, image_file_name + '.txt' )

                # Given the fact that annotations are to be present as agreed on earlier we check for annotation data

                annot_parser = AnnotParser()

                if isfile(txt_annotation_file): # pottential GT in txt format
                    
                    ground_truths = annot_parser.from_txt( txt_annotation_file, self.params['MODEL_INPUT_WIDTH'], self.params['MODEL_INPUT_HEIGHT'] )

                # json annots take precedence over the txt annots
                
                if isfile(json_annotation_file): # pottential GT in json format

                    ground_truths = annot_parser.from_json(json_annotation_file, self.params['TRAIN_INFO']['CLASS_NAMES'])

                    if ground_truths is None: 

                        print('Skipping ', image_file, 'since it has isBadImage flag')
                        continue

                negative_example = False
                
                if np.sum(ground_truths[0]) == 0: # I.E. negative example [ [0, 0, 0, 0, 0, 0] ]
                    
                    negative_example = True
                    
                    if abs_bboxes:
                    
                        pred_dir = 'false_positives'
                        
                    else:
                        
                        pred_dir = 'true_negative'
                
                if not negative_example:
                    
                    pred_dir = 'true_positives'
                    is_false_positive = False
                    
                    ious = []
                    for ground_truth in ground_truths:
                        
                        if is_false_positive: break
                        
                        if len(abs_bboxes) == 0:
                            
                            pred_dir = 'false_negative'
                            break
                        
                        for abs_bbox in abs_bboxes:
                            
                            iou = bbox_iou(ground_truth, abs_bbox)
                            ious.append(iou)
                        
                        for iou in ious:
                            
                            if iou < self.params['YOLO_SPEC']['IOU_THRESHOLD']:
                                
                                pred_dir = 'false_positives'
                                is_false_positive = True
                            
                annotated_image = self._draw_boxes(image, ground_truths, color = (0, 0, 255), y_shift = 20)

            pred_dir = join(dst_dir, pred_dir)
            if not isdir(pred_dir): mkdir(pred_dir)
                
            dst_file = join(pred_dir, basename(image_file))
            
            if isfile(dst_file):
                
                print('Oh no, just overriden an existing file with the name ', dst_file)
            
            annotated_image = resize(annotated_image, (400, 400))
            flag = imwrite(dst_file, annotated_image)

        print('Finished, predictions can be found at', dst_dir)

