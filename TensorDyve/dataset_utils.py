import numpy as np
import tensorflow as tf

from copy import deepcopy
from multiprocessing.dummy import Process

from itertools import zip_longest
from os import listdir, mkdir
from os.path import join, basename, dirname, isfile, isdir, splitext
from sys import stdout
from time import sleep, time

from cv2 import imread, IMREAD_COLOR, IMREAD_GRAYSCALE, imshow, waitKey, resize
from psutil import virtual_memory
from tqdm import tqdm

from glob import glob
from json import load, dump
from json.decoder import JSONDecodeError
from multiprocessing import cpu_count

from .custom_exceptions import MemThresholdReached, TrainEvalDatasetFormatMismatch

from .utils import LOG

from .project_globals import ANNOTATION_DS_FLAG_KEY, TF_VERSION_KEY, \
    CLASS_NAMES_KEY, NUM_CLASSES_KEY, NUM_TRAIN_EXAMPLES_KEY, NUM_EVAL_EXAMPLES_KEY, \
    JSON_ANNOTATION_DIR, JSON_ANNOT_KEY, JSON_LABEL_KEY, JSON_TOP_LEFT_KEY, \
    JSON_TOP_RIGHT_KEY, JSON_BOTTOM_LEFT_KEY, JSON_BOTTOM_RIGHT_KEY, \
    JSON_IS_BAD_IMAGE_KEY, TRAIN_TFR_NAME, EVAL_TFR_NAME, METADATA_FILE_NAME, \
    TFR_IMPLICIT_DIR_NAME, TFR_SHAPE_KEY, TFR_SAMPLE_KEY, TFR_GROUND_TRUTH_KEY, ALLOWED_EXTENSIONS


class AnnotParser():

    # might get removed since there is no use for having datasets in this format, better to convert it to json prior, code gets messy because of this
    def from_txt(self, file_):

        image_annotations = []

        with open(file_, 'r') as handle:
            
            lines = handle.readlines()
            
        content = [ line.strip() for line in lines ]
        
        for annotation in content:
            
            data = annotation.split()
            
            x = int(float(data[1]))
            y = int(float(data[2]))
            w = int(float(data[3]))
            h = int(float(data[4]))

            xmin = x - w // 2
            xmax = x + w // 2
            ymin = y - h // 2
            ymax = y + h // 2
            
            cls_index = int(data[0])
        
            image_annotations.append([ xmin, ymin, xmax, ymax, 1, cls_index ])
        
        if len(image_annotations) == 0: # I.E. negative example
            
            image_annotations = [[0, 0, 0, 0, 0, 0]] # for consistency, the label for a negative example will be this, even tho it is ONE annotation.
    
        return image_annotations

    def from_json(self, file_, im_w, im_h, class_names):

        image_annotations = []
        
        with open(file_, 'r') as handle:

            json_data = load(handle)

        is_bad_image = json_data.get(JSON_IS_BAD_IMAGE_KEY, False)
    
        if is_bad_image:
            
            return None
            
        else:
            
            annot_dicts = json_data[JSON_ANNOT_KEY]
            
            for annot_dict in annot_dicts:
                
                class_label = annot_dict[JSON_LABEL_KEY]
                
                if class_label == '':
                    
                    class_label = '0'
                                    
                cls_index = class_names.index(class_label)
        
                tl_xy_string = annot_dict[JSON_TOP_LEFT_KEY]
                tr_xy_string = annot_dict[JSON_TOP_RIGHT_KEY]
                bl_xy_string = annot_dict[JSON_BOTTOM_LEFT_KEY]
                br_xy_string = annot_dict[JSON_BOTTOM_RIGHT_KEY]
                
                x_string, y_string = tl_xy_string.split(',')
                tl_x, tl_y = float(x_string), float(y_string)
                
                x_string, y_string = tr_xy_string.split(',')
                tr_x, tr_y = float(x_string), float(y_string)
                
                x_string, y_string = bl_xy_string.split(',')
                bl_x, bl_y = float(x_string), float(y_string)
                
                x_string, y_string = br_xy_string.split(',')
                br_x, br_y = float(x_string), float(y_string)
    
                xmin = np.min( [ tl_x, tr_x, bl_x, br_x ] )
                xmax = np.max( [ tl_x, tr_x, bl_x, br_x ] )

                ymin = np.min( [ tl_y, tr_y, bl_y, br_y ] )
                ymax = np.max( [ tl_y, tr_y, bl_y, br_y ] )
                
                # filling in relative coords so that a resize can happen in the input pipeline without any fuss related to keypoints adjustment
                
                image_annotations.append( [ xmin / im_w, ymin / im_h, xmax / im_w, ymax / im_h, 1 , cls_index ] ) # obj_prob filled in there for consistency
    
        if len(image_annotations) == 0: # I.E. negative example
            
            image_annotations = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] # for consistency, the label for a negative example will be this, even tho it is ONE annotation.
    
        return image_annotations

class ImgDataset():
    
    '''
    
        Info:
        
            This class was designed to be some sort of handle for the supported dataset standards: 
                Dyve classifier, 
                Darknet detector, 
                DyveAnnotationTool detector
            
            If the dataset is to be used concretly, one can make use of this instance in order to 
            
                fetch file paths to read images, 
                fetch file paths in order to get boxes from them,
                build one-hot encodings for the integer labels associated with one particular image file,
                etc ...
            
    '''
        
    def __init__(self, color, memory_threshold, ds_dir = None, split_percent = None, tr_dir = None, ev_dir = None):
    
        self._color = color
        self._memory_threshold = memory_threshold

        self._ds_dir = ds_dir
        self._split_percent = split_percent
        self._tr_dir = tr_dir
        self._ev_dir = ev_dir
        
        self._class_name_blacklist = [TFR_IMPLICIT_DIR_NAME, 'split'] # it's convenient to have let's say a split_dir where you split your dataset right next to the current dataset
        self._imread_mode = IMREAD_COLOR if color else IMREAD_GRAYSCALE
        
        # populated after call to self._build()
        
        self._annot_ds_flag = None
        
        self._num_classes = 0
        self._class_names = []
        
        self._train_samples = []
        self._train_ground_truths = [] # might have to remove these since the information about labeling is encoded in self._train_samples
        self._eval_samples = []
        self._eval_ground_truths = [] # might have to remove these since the information about labeling is encoded in self._eval_samples
    
        self._num_train_examples = 0
        self._num_eval_examples = 0
        
        self._build()

    def is_annotation_dataset(self):
        
        return self._annot_ds_flag
    
    def get_tr_dir(self):
        
        return self._tr_dir
    
    def get_ev_dir(self):
        
        return self._ev_dir
    
    def get_ds_dir(self):
        
        return self._ds_dir
    
    def get_num_classes(self):
        
        return self._num_classes
    
    def get_num_train_examples(self):
        
        return self._num_train_examples
    
    def get_num_eval_examples(self):
        
        return self._num_eval_examples
    
    def get_class_names(self):
        
        return self._class_names
    
    def get_dataset_metadata(self):
        
        return {
        
                    NUM_CLASSES_KEY        : self._num_classes,
                    NUM_TRAIN_EXAMPLES_KEY : self._num_train_examples,
                    NUM_EVAL_EXAMPLES_KEY  : self._num_eval_examples,
                    ANNOTATION_DS_FLAG_KEY : self._annot_ds_flag,
                    TF_VERSION_KEY         : tf.__version__,
                    CLASS_NAMES_KEY        : self._class_names
                        
            }
      
    def get_train_split(self):
        
        return self._train_samples, self._train_ground_truths
      
    def get_eval_split(self):
        
        return self._eval_samples, self._eval_ground_truths  
    
        
    def _fetch_annotation_dataset(self, all_samples, annot_dir, json_annot_flag):


        '''
        
            TODO: evenly distribute workload over cpu cores
        
        '''

        all_ground_truths = []
        class_names = set()
            
        if json_annot_flag:
                
            LOG('Scaning for valid .json annotation files ...')
            
            for sample in tqdm(all_samples):
                    
                file_name = basename(sample)
                file_name = splitext(file_name)[0]
                annot_file = join( annot_dir, file_name + '_forTraining.json' )
                    
                if isfile(annot_file):
                        
                    all_ground_truths.append(annot_file)
                        
                    with open(annot_file, 'r') as handler:
                    
                        json_data = load(handler)
                    
                    annotations = json_data[JSON_ANNOT_KEY]
                        
                    for annot in annotations:
                        
                        class_string_label = annot[JSON_LABEL_KEY]
                            
                        if class_string_label == '':
                                
                            class_string_label = '0'
                        
                        set(class_dir_names).add( class_string_label )
                        
                else:
                        
                    all_ground_truths.append(None) 
            
            LOG('Done !')
                
        else:
                
            LOG('Scanning for valid .txt annotation files ...')
            
            for sample in tqdm(all_samples):
                    
                file_name = basename(sample)
                file_name = splitext(file_name)[0]
                annot_file = join(annot_dir, file_name + '.txt')
                    
                if isfile(annot_file):
                        
                    all_ground_truths.append(annot_file)
                        
                    with open(annot_file, 'r') as handler:
            
                        lines = handler.readlines()
                            
                    content = [ line.strip() for line in lines ]
                        
                    for annotation in content:
                            
                        data = annotation.split()
                  
                        set(class_dir_names).add( data[0] )
                
                else:
                        
                    all_ground_truths.append(None)

            LOG('Done !')

        # have to batch up both samples and gts this way due to some inconsistency in self._build cause by the split method
        return [all_samples], [all_ground_truths], list(all_class_string_labels)
    
    def _infer_dataset_type(self, root_ds_dir):
        
        all_samples = []
        all_labels = []
    
        json_annots_flag = False
        
        '''
            Note:
             
                'class_string_names' is to be a set, containing the class names, unsorted, in a potentially split dataset
         
            Note:
               
                this will fail if some images are placed alongside the class directories if the dataset is intended for a classifier
                since it will think the dataset is intended for either darnet format or dyve annotation tool format
        
        '''   
            
        total_num_examples = 0
        
        for ext in ALLOWED_EXTENSIONS: 
            
            all_samples += glob( join( root_ds_dir, '*' + ext) )
        
        total_num_examples = len(all_samples)
        
        if total_num_examples == 0:
            
            annot_ds_flag = False
    
            all_samples, all_labels, class_string_names, total_num_examples = self.load_dir( root_ds_dir )
            
        else:
            
            annot_ds_flag = True
            
            annot_dir = root_ds_dir
            
            _txt_annot = glob( join( annot_dir, '*' + '.txt' ) )
            
            json_annots_flag = False
            
            if not _txt_annot:
                
                json_annots_flag = True
                
                annot_dir = join( root_ds_dir, JSON_ANNOTATION_DIR )
                
                _json_annot = glob( join( annot_dir, '*' + '.json' ) )
                
                if not _json_annot:
    
                    raise ValueError('Invalid dataset type.')

            all_samples, all_labels, class_string_names = self._fetch_annotation_dataset(all_samples, annot_dir, json_annot_flag)

        return all_samples, all_labels, class_string_names, annot_ds_flag, total_num_examples
     
    
    def unison_shuffle(self, samples, labels):
        
        '''
        
            TODO: evenly distribute workload over cpu cores
        
        '''
        
        shuffled_samples = []
        shuffled_labels = []
        
        max_index = len(samples)
        
        indices = np.random.permutation(max_index)

        for candidate_index in indices:
            shuffled_samples.append(samples[candidate_index])
            shuffled_labels.append(labels[candidate_index])

        return shuffled_samples, shuffled_labels
    
    def split(self, all_class_samples, all_class_labels):
        
        '''
        
            TODO: evenly distribute workload over cpu cores
        
        '''
        
        tr_samples = []
        tr_labels = []
        ev_samples = []
        ev_labels = []
        
        num_train_examples, num_eval_examples = 0, 0
        
        LOG('Splitting image dataset ...')
        for samples, labels in zip(tqdm(all_class_samples), all_class_labels) :
            
            total = len(samples)
            split_index = int(total * self._split_percent)
        
            ith_class_tr_samples = samples[ 0: split_index ]
            tr_samples.append( ith_class_tr_samples )
            num_train_examples += len(ith_class_tr_samples)
            
            ith_class_eval_samples = samples[ split_index : total ]
            ev_samples.append( ith_class_eval_samples )  
            num_eval_examples += len(ith_class_eval_samples)
             
            tr_labels.append( labels[ 0 : split_index ] ) 
            ev_labels.append( labels[ split_index: total ] ) 

        LOG('Done !')
        return tr_samples, tr_labels, num_train_examples, ev_samples, ev_labels, num_eval_examples
    
    
    def find_sub_dir_paths(self, root_dir):
       
        _class_dir_names = listdir(root_dir)
        class_dir_names = []
        
        for class_dir_name in _class_dir_names:
            
            if isdir(join(root_dir, class_dir_name)):
                
                class_dir_names.append(class_dir_name)
        
        for blacklist_entry in self._class_name_blacklist:
        
            if blacklist_entry in class_dir_names:
            
                class_dir_names.remove( blacklist_entry )
        
        return [ join(root_dir, class_dir_name) for class_dir_name in class_dir_names ], class_dir_names
    
    def load_dir(self, root_dir):
        
        '''
        
            TODO: evenly distribute workload over cpu cores
        
        '''
        
        class_dir_paths, class_dir_names = self.find_sub_dir_paths(root_dir)
        
        num_classes = len(class_dir_names)
        
        samples = []
        labels = []
        
        LOG('Fetching image file paths ...')
        
        total_num_examples = 0
        
        '''
        
            
        
        
        '''
        
        
        for i in tqdm(range(num_classes)):
            
            path = class_dir_paths[ i ]
            
            ith_class_samples = []
            
            for ext in ALLOWED_EXTENSIONS: 
                
                ith_class_samples += glob(join(path, '*' + ext))
            
            nr_samples = len(ith_class_samples)
            total_num_examples += nr_samples
            
            ith_class_labels = [i] * nr_samples
            
            samples.append(ith_class_samples)
            labels.append(ith_class_labels)

        LOG('Done !')
        return samples, labels, class_dir_names, total_num_examples
    
    
    def get_image(self, path):
    
        img = imread(path, self._imread_mode)
        
        if not self._color:
            
            img = np.expand_dims(img, 2)
    
        return img

    def get_ground_truth(self, ground_truth, im_w, im_h):

        if not self._annot_ds_flag:
            
            return ground_truth
        
        else:
            
            # this might be a mistake
            if ground_truth is None: 
                
                return [ [0, 0, 0, 0, 0, 0] ]
            
            ext = splitext(ground_truth)[1].lower()
            
            annot_parser = AnnotParser()
            
            if ext == '.txt':
                
                return annot_parser.from_txt(ground_truth)
            
            elif ext == '.json':
            
                return annot_parser.from_json(ground_truth, im_w, im_h, self._class_names) # passing in imw and imh to convert to relative coords, so that when deserializing happens, possible resize, easy adjustment for boxes
            
            else:
                
                msg = 'Invalid annotation file format: ', ext, 'not supported.'
                raise ValueError(msg)
            
 
    def load_in_memory(self):
    
        
        def _load_loop(samples, gts):
            
            images, ground_truths = [], []
            
            for ith_class_samples, ith_class_gts in zip(tqdm(samples), gts):
                
                for sample, gt in zip(ith_class_samples, ith_class_gts):
            
                    image = self.get_image(sample)
                    h, w = image.shape[:2]
                    ground_truth = self.get_ground_truth(gt, w, h)
                
                    if ground_truth is None: continue
                
                    images.append(image)
                    ground_truths.append(ground_truth)
                
                    if virtual_memory().percent > self._memory_threshold:
                
                        msg = 'Failed to load dataset into memory because it exceeds the user defined memory threshold of ' + str(self._memory_threshold) + ' percent'
                        raise MemThresholdReached(msg)
        
            return images, ground_truths
        
        LOG('Attempting to load train dataset in memory ...')
        
        train_images, train_ground_truths = _load_loop(self._train_samples, self._train_ground_truths)
        
        LOG('Done !')
        
        
        LOG('Attempting to load eval dataset in memory ...')
        
        eval_images, eval_ground_truths = _load_loop(self._eval_samples, self._eval_ground_truths)
        
        LOG('Done !')
        
        return (
            
            np.asarray(train_images, dtype=np.uint8),
            np.asarray(train_ground_truths, dtype=np.uint8),
            np.asarray(eval_images, dtype=np.uint8),
            np.asarray(eval_ground_truths, dtype=np.uint8) )


    def _flatten_2d_py_list(self, to_flat):
        
        return [ item for sublist in to_flat for item in sublist ]
    
    def _build(self):

        # Note: self._infer_dataset_type throws exception if the dataset format is not supported

        if self._ds_dir is not None:

            LOG('Infering image dataset type ...')
            all_samples, all_ground_truths, all_class_names, annot_ds_flag, total_num_examples = self._infer_dataset_type(self._ds_dir) 
            LOG('Done !')

            ( self._train_samples, self._train_ground_truths, self._num_train_examples, 
                self._eval_samples, self._eval_ground_truths, self._num_eval_examples ) = self.split(all_samples, all_ground_truths)
        
        else: # I.E split has been made prior to invoking utility
            
            LOG('Infering train image dataset type ...')
            self._train_samples, self._train_ground_truths, tr_class_names, tr_annot_ds_flag, self._num_train_examples = self._infer_dataset_type(self._tr_dir)
            LOG('Done !')
            
            LOG('Inferring eval image dataset type ...')
            self._eval_samples, self._eval_ground_truths, ev_class_names, ev_annot_ds_flag, self._num_eval_examples = self._infer_dataset_type(self._ev_dir)
            LOG('Done !')
            
            if len(tr_class_names) != len(ev_class_names):
                
                print('Apparently the number of classes found in the train set is differnt than the one found in the evaluaton set ...')
                print('Train set class names:', tr_class_names)
                print('Eval set class names:', ev_class_names)
                print('Just a short notice, you might wanna double check ...')
                
                cnt = 3
                for i in range(cnt):
                    
                    print('Resuming training in', cnt - i, '...')
                    sleep(1)
            
            if tr_annot_ds_flag != ev_annot_ds_flag: 
                
                def _helper(tr_type, ev_type):
                    
                    return 'One cannot train a model on a ' + tr_type + ' dataset and evaluate it on a ' + ev_type + ' dataset' 
                
                if tr_annot_ds_flag: msg = _helper('classification', 'detection')
                if ev_annot_ds_flag: msg = _helper('detection', 'classification')
                    
                '''
                    
                    Note:
                        
                        Maybe this is to be implemented in the future as a conventient way to work with the two formats seprately
                        since the utility supports training a classifier 
                        from a dataset with annotations boxes that are to be cropped alongside the standard Dyve classifier dataset format.
                
                '''
                    
                raise TrainEvalDatasetFormatMismatch(msg)
            
            # since there is no splitting to be done, the per class grouping as [[c1_sample1, c1_sample1...], [c2_sample1..., c2_sample2], ..] has to go away
            
            all_class_names = set()
            
            all_class_names = all_class_names.union(tr_class_names)
            all_class_names = all_class_names.union(ev_class_names)
            
            annot_ds_flag = tr_annot_ds_flag and ev_annot_ds_flag # a bit redundant but hey, because too cautious
                        
        '''
        
            TODO:
            
                Get statistics about the number of images per class, 
                so that one can make use of this to either dump in tfr metadata or even in train info config file.
        
        '''
        
        _class_names = list(all_class_names)
        _num_classes = len(_class_names)
            
        self._class_names = sorted(_class_names)
        self._num_classes = _num_classes

        self._annot_ds_flag = annot_ds_flag
        
        ''' currently will experiment with 'tfr per class grouping, where each class tfr will be split into shards', this shuffling is not needed
        
        LOG('Shuffling train set ...')
        self._train_samples, self._train_ground_truths = self.unison_shuffle(tr_samples, tr_labels)
        LOG('Done !')
        
        LOG('Shuffling eval set ...')
        self._eval_samples, self._eval_ground_truths = self.unison_shuffle(ev_samples, ev_labels)
        LOG('Done !')
        
        '''
        
class TFRGenerator():
    
    def __init__(self, img_dataset, num_tfr_shards, tfr_destination_directory):
        
        self.img_dataset = img_dataset
            
        self.tfr_root_dir = join(tfr_destination_directory, TFR_IMPLICIT_DIR_NAME)
        
        self.num_tfr_shards = num_tfr_shards 
        
        self._ground_truth_dtype = np.float32
        self._shape_dtype = np.int32
        
    def int64_feature(self, value):
        
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[ value ]))
    
    def bytes_feature(self, value):
        
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ value ]))
     
    def build_feature(self, image, ground_truths):
        
        if self.img_dataset.is_annotation_dataset():
     
            nr_annotations = len(ground_truths)
            
            annots = []
            
            for gt in ground_truths:
                
                annots += gt
            
            annots += [nr_annotations]
            
            '''
                
                flattened version of the original [ [xmin, xmax, ymin, ymax, cls_index], [xmin, xmax, ymin, ymax, cls_index], ... ] with the last element 
                being the number of annotaitons, which is to be poped after deserialization in order to reshape the array to it's original shape.
            
            '''
            
            ground_truths = np.asarray(annots, dtype = self._ground_truth_dtype)
            
            tfr_ground_truth_feature = self.bytes_feature(tf.compat.as_bytes(ground_truths.tostring()))
        
        else:
            
            tfr_ground_truth_feature = self.int64_feature(ground_truths)
        
        image_shape = np.asarray( image.shape, dtype = self._shape_dtype )
        
        return {
            
            TFR_SHAPE_KEY : self.bytes_feature(tf.compat.as_bytes( image_shape.tostring() ) ),
            TFR_SAMPLE_KEY : self.bytes_feature(tf.compat.as_bytes(image.tostring())),
            TFR_GROUND_TRUTH_KEY  : tfr_ground_truth_feature
        }
     
     
    def export_dataset_metadata(self, entry):
        
        metadata_file = join(self.tfr_root_dir, METADATA_FILE_NAME)
        
        if isfile(metadata_file):
            
            with open(metadata_file, 'r') as json_file:
                
                json_data = load(json_file)
                json_data.update(entry)
                
            with open(metadata_file, 'w') as json_file:
                
                dump(json_data, json_file, indent=10)    
            
        else: 
            
            json_data = entry
            
            with open(metadata_file, 'w') as json_file:
                
                dump(json_data, json_file, indent=10)
    
    def _convert(self, all_samples, all_ground_truths, split_name):
        
        # Will add images with no annotations als, they count as negative examples and because of this, in the input pipe, if it is intended
        # for a classifier, skip the ones with 0 annotations
        
        max_workers = min(cpu_count(), self.num_tfr_shards)
        
        sample_workloads, gt_workloads = [ [] for i in range(max_workers) ], [ [] for i in range(max_workers) ]
        
        worker_index = 0
        
        _all_samples, _all_ground_truths = [], []
        
        for ith_class_samples, ith_class_gts in zip(all_samples, all_ground_truths):
            
            _all_samples += ith_class_samples
            _all_ground_truths += ith_class_gts
        
        shuffled_all_samples, shuffled_all_gts = self.img_dataset.unison_shuffle(_all_samples, _all_ground_truths) 
        
        for sample, gt in zip(shuffled_all_samples, shuffled_all_gts):
        
            sample_workloads[worker_index] += [sample]
            gt_workloads[worker_index] += [gt]
        
            worker_index += 1
            if worker_index == max_workers: worker_index = 0
            
        def _write_shards(sample_workload, gt_workload, split_name, id):
            
            _sample_workload = tqdm(sample_workload)
            _sample_workload.set_description('Threads ' + str(id) + ' progress:')
            
            num_worker_shards = self.num_tfr_shards // max_workers
            num_examples = len(sample_workload)
            num_examples_per_shard = max(num_examples // num_worker_shards, 100) # bounding lower limit for examples per tfr shard to be 100
 
            new_shard = False
            
            example_counter, shard_count = 0, 0
                
            writer = tf.python_io.TFRecordWriter(join(self.tfr_root_dir,  str(id) + '_shard_' + str(shard_count) + '_' + split_name))
            
            for sample, gt in zip(_sample_workload, gt_workload):
                
                if new_shard:
                        
                    writer = tf.python_io.TFRecordWriter(join(self.tfr_root_dir,  str(id) + '_shard_' + str(shard_count) + '_' + split_name))
                    new_shard = False
                    
                image = self.img_dataset.get_image(sample)
                    
                h, w = image.shape[:2]
                    
                gt = self.img_dataset.get_ground_truth(gt, w, h)
                    
                feature = self.build_feature(image, gt)
                    
                example = tf.train.Example( features = tf.train.Features( feature = feature ) )
        
                writer.write(example.SerializeToString())
  
                example_counter += 1
                if example_counter == num_examples_per_shard:
                        
                    example_counter = 0 
                    shard_count += 1
                    new_shard = True
                              
        procs = []
        
        id = 1
        for sample_workload, gt_workload in zip(sample_workloads, gt_workloads):
            
            procs.append(
                
                Process(
                    
                    target = _write_shards, 
                    args = (sample_workload, gt_workload, split_name, id)
                    
                )
            )
        
            id += 1
        
        for p in procs: p.start()
            
        for p in procs: p.join()
  
              
    def convert(self):

        if not isdir(self.tfr_root_dir): mkdir(self.tfr_root_dir)

        start = time()
        
        LOG('Serializing train dataset to ' + TRAIN_TFR_NAME)
        tr_samples, tr_gts = self.img_dataset.get_train_split()
        self._convert(tr_samples, tr_gts, TRAIN_TFR_NAME)
        LOG('Finished writing train tfrs in: ' + str(time() - start) + ' seconds')
        
        start = time()
        
        LOG('Serializing eval dataset to ' + EVAL_TFR_NAME)
        ev_samples, ev_gts = self.img_dataset.get_eval_split()
        self._convert(ev_samples, ev_gts, EVAL_TFR_NAME)
        LOG('Finished writing evaluation tfr in: ' + str(time() - start) + ' seconds')
    
        LOG('TFR dataset found in directory: ' + self.tfr_root_dir + ' alongside the dataset metadata file.')
        
        self.export_dataset_metadata( self.img_dataset.get_dataset_metadata() )
        
        return self.tfr_root_dir
        
