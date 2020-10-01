import tensorflow as tf

import numpy as np

import TensorDyve

from sys import float_info
from os.path import join, isdir, isfile, basename
from os import mkdir, remove
from glob import glob
from copy import copy
from shutil import copyfile
from re import split
from json import load, dump

from time import time

from .project_globals import BEST_CKPT_DIR_NAME, BEST_METRICS_BUFFER_NAME
from .project_globals import BEST_TRAIN_IOU_KEY, BEST_EVAL_IOU_KEY, BEST_TRAIN_F1_SCORE_KEY, BEST_EVAL_F1_SCORE_KEY
from .project_globals import BEST_TRAIN_ACC_KEY, BEST_EVAL_ACC_KEY

from .custom_exceptions import NoProgress

###########################
current_train_mean_iou = -1
current_train_precision = -1
current_train_recall = -1
current_train_f1_score = -1
      
current_eval_mean_iou = -1
current_eval_precision = -1
current_eval_recall = -1
current_eval_f1_score = -1
        
best_train_mean_iou = -1
best_train_precision = -1
best_train_recall = -1
best_train_f1_score = -1

best_eval_mean_iou = -1
best_eval_precision = -1
best_eval_recall = -1
best_eval_f1_score = -1
###########################        

red_flag_count = 0
eval_acc_value_history = []

######################
current_train_acc = -1
current_eval_acc = -1

best_train_acc = -1
best_eval_acc = -1
######################

'''

    TODO:
    
        major refactor, redudancy in implementation and thus usability


'''


class _SessionRunHOOK(tf.train.SessionRunHook):

    def __init__(self, loss_op, reg_loss_op, total_loss_op, global_step_op, logdir, tolerance_count, train_flag, hide_meta):
        
        self.loss_op = loss_op
        self.reg_loss_op = reg_loss_op
        self.total_loss_op = total_loss_op
        self.global_step_op = global_step_op
        self.tolerance_count = tolerance_count
        
        self.learning_rate_tensor = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        
        sub_dir = join('logs', 'train') if train_flag else join('logs', 'eval')
        self.summary_writer = tf.summary.FileWriter( join(logdir, sub_dir), flush_secs = 0)
        
        self.train_flag = train_flag
        self.hide_meta = hide_meta
        
        self.mode_string = 'Train' if train_flag else 'Evaluation'

        self.logdir = logdir

        self.cumulative_total_loss = 0
        self.cumulative_loss = 0
        self.cumulative_reg_loss = 0
        self.seen_batches = 0
        
        self._decimal_precision = 5 # to be used in order to prevent any infinitecimal increment that might be considered as a platteau, when training time is expensive 


    def clear_old_ckpts(self, ckpt_dir):
        
        print('Clearing older checkpoints in best checkpoints dir ...')
        
        all_files = glob(join(ckpt_dir, '*'))
        
        for file in all_files:
                
                remove(file)

    def begin(self):
        
        print('Starting ' + self.mode_string + ' session')

class ClassifierSessionRunHook(_SessionRunHOOK):

    def __init__(self,
        
        acc_tensor,
        acc_update_op,
        
        loss_op,
        reg_loss_op, 
        total_loss_op,
        global_step_op,
        logdir,
        tolerance_count,
        train_flag,
        hide_meta
        
    ):
    
        super().__init__(loss_op, reg_loss_op, total_loss_op, global_step_op, logdir, tolerance_count, train_flag, hide_meta)

        self.best_metrics_buffer = join( join(self.logdir, BEST_CKPT_DIR_NAME), BEST_METRICS_BUFFER_NAME )

        if isfile( self.best_metrics_buffer ):
            
            global best_train_acc
            global best_eval_acc
            
            with open( self.best_metrics_buffer, 'r' ) as handle:
            
                _dict = load( handle )
                
                best_train_acc = float( _dict[ BEST_TRAIN_ACC_KEY ] ) 
                best_eval_acc = float( _dict[ BEST_EVAL_ACC_KEY ] )

        self.acc_tensor = acc_tensor
        self.acc_update_op = acc_update_op

        ''' tracking batch norm values

        self.all_batch_norm_var_ids = []
        self.all_batch_norm_vars = []

        self.bn_log_dict = dict()

        _graph = tf.get_default_graph()

        keys = _graph.get_all_collection_keys()

        signatures = ['gamma', 'beta', 'moving_mean', 'moving_variance']

        for key in keys:

            # print('fetching collection', key)

            elements = _graph.get_collection_ref(key)

            for element in elements:

                if 'bn_1' in element.name:

                    for signature in signatures:

                        if signature in element.name:                   

                            self.all_batch_norm_vars.append(element)
                            self.all_batch_norm_var_ids.append(element.name)

        '''

    def save_best_checkpoint(self, session, global_step):

        global current_train_acc
        global current_eval_acc
        global best_train_acc
        global best_eval_acc
        
        global red_flag_count
        global eval_acc_value_history

        #print("Current eval acc {0:.15f}".format(current_eval_acc))
        #print("Best eval acc before possible update, {0:.15f}".format(best_eval_acc))
        
        _current_eval_accuracy = round(current_eval_acc, self._decimal_precision)
        
        # there are some inconsistencies in terms of rounding up either when operating with floating point numbers or doing assignments
        # comparison between rounded values in order to neglect any infinitecimal increments when platteau is to be detected when training is expensive
        _better_evaluation = _current_eval_accuracy > round(best_eval_acc, self._decimal_precision) 
        
        if self.tolerance_count > 0:
            
            if not _better_evaluation:
            
                red_flag_count += 1
                eval_acc_value_history.append(_current_eval_accuracy)
            
            else:
            
                red_flag_count = 0
                eval_acc_value_history = []

            if red_flag_count == self.tolerance_count:
            
                message = ( 
             
                    'Training stagnated or diverged with respect to the evaluation accuracy ' + str(self.tolerance_count) + 
                    ' times in a row, here are the trailing accuracy values for revision ...' + '\n' +
                    str(eval_acc_value_history) + '\n' +
                    'last best recorded accuracy on the evaluation set is ' + str(best_eval_acc) + '\n' +
                    'Training has ended.'
                )
            
                print(message)
                exit(0)

        better_evaluation = current_eval_acc > best_eval_acc
        
        if ( better_evaluation or ( (current_eval_acc == 1) and (current_train_acc > best_train_acc) ) ):
            
            print('Found a better set of weights for the model ...')
           
            best_ckpt_dir = join(self.logdir, BEST_CKPT_DIR_NAME)
            if not isdir(best_ckpt_dir): mkdir(best_ckpt_dir)

            self.clear_old_ckpts(best_ckpt_dir)

            saver = session.graph.get_collection('savers')[0]
            
            saver.save(
                
                session,
                join( best_ckpt_dir, 'model.ckpt' ),
                global_step = global_step,
                latest_filename = None,
                meta_graph_suffix = 'meta',
                write_meta_graph = not self.hide_meta,
                write_state = True,
                strip_default_attrs = False
            )
            
            best_train_acc = current_train_acc
            best_eval_acc  = current_eval_acc
            
            #print("Best eval acc afters possible update, {0:.15f}".format(best_eval_acc))
            
            TensorDyve.tensor_dyve.best_train_acc = best_train_acc
            TensorDyve.tensor_dyve.best_eval_acc = best_eval_acc

            with open(self.best_metrics_buffer, 'w') as handle:
                
                dump({BEST_TRAIN_ACC_KEY: str(best_train_acc), BEST_EVAL_ACC_KEY: str(best_eval_acc) }, handle, indent = 5)

    def before_run(self, run_context):
        
        base_ = [
             
            self.acc_update_op, # forcing metric_update_op to be executed after each batch in call to estimator.train
            self.loss_op,
            self.reg_loss_op,
            self.total_loss_op,
            self.learning_rate_tensor,
            tf.train.get_global_step()
             
        ]

        return tf.train.SessionRunArgs( 
        
            base_ # + [self.all_batch_norm_vars] # tracking batch norm values
                                       
        )

    def after_run(self, run_context, run_values):
        
        ''' tracking batch norm values
        
        batch_mean_accuracy, batch_loss, batch_reg_loss, batch_total_loss, learning_rate, global_step, all_batch_norm_var_values = run_values.results
        
        '''

        batch_mean_accuracy, batch_loss, batch_reg_loss, batch_total_loss, learning_rate, global_step = run_values.results
        
        if batch_mean_accuracy is None or batch_loss is None or batch_reg_loss is None or batch_total_loss is None or learning_rate is None or global_step is None:
          print('Invalid statistics: ', batch_mean_accuracy, batch_loss, batch_reg_loss, batch_total_loss, learning_rate, global_step)
          return

        log_message = 'Mean accuracy this session so far: {}'.format(batch_mean_accuracy) # this is on a per batch basis
        if self.train_flag:
            log_message = log_message + ', training step: {}'.format(global_step)

        print(log_message)

        self.cumulative_loss += batch_loss
        self.cumulative_reg_loss += batch_reg_loss
        self.cumulative_total_loss += batch_total_loss
        self.seen_batches += 1

        ''' tracking batch norm values

        timestamp = time()
        
        self.bn_log_dict[ timestamp ] = dict()

        for batch_norm_var_value, batch_norm_var_id in zip(all_batch_norm_var_values, self.all_batch_norm_var_ids):

            _tempo = []
            
            for _val in batch_norm_var_value:

                _tempo.append( float(_val) ) # json cannot serialize numpy arrays or even numpy scalars, so casting is required at an element level to end up with python lists of python floats

            self.bn_log_dict [ timestamp ] [ batch_norm_var_id ] = _tempo

        '''

        if self.train_flag: # no need to log learning rate if evaluation is going on
            
            summary = tf.Summary()
            summary.value.add( tag = 'LearningRate', simple_value = learning_rate )
        
            self.summary_writer.add_summary( summary, global_step )
            self.summary_writer.flush()
    
    def end(self, session):

        ''' tracking batch norm values

        bn_log_file = join( self.logdir, 'train_bn.json' if self.train_flag else 'eval_bn.json' )

        with open(bn_log_file, 'a') as fd:

            dump(self.bn_log_dict, fd, indent = 3, sort_keys = True)

        '''

        global current_train_acc
        global current_eval_acc

        '''
            NOTE:
                during the session the state of the mean_iou metric is update with the update op, 
                at the end, given the state, evaluate the metric tensor and get the mean_iou for that particlar session
        '''
        mean_accuracy_value, global_step = session.run( [ self.acc_tensor, self.global_step_op ] ) 
        mean_loss_value = self.cumulative_loss / self.seen_batches
        mean_reg_loss_value = self.cumulative_reg_loss / self.seen_batches
        mean_total_loss_value = self.cumulative_total_loss / self.seen_batches 
       
        print('------------------------------------------------------------------------------')
        
        print('END of ', self.mode_string, 'session')
        
        print('Mean Accuracy: ', mean_accuracy_value)
        print('Mean Total LOSS: ' , mean_total_loss_value)
        
        print('------------------------------------------------------------------------------')
        
        summary = tf.Summary()
        
        summary.value.add( tag = 'Accuracy', simple_value = mean_accuracy_value )
        
        summary.value.add( tag = 'Loss/Standalone', simple_value = mean_loss_value )
        summary.value.add( tag = 'Loss/Regularization', simple_value = mean_reg_loss_value )
        summary.value.add( tag = 'Loss/Total', simple_value = mean_total_loss_value )
        
        self.summary_writer.add_summary( summary, global_step )
        self.summary_writer.flush()
        
        if self.train_flag:
            
            current_train_acc = mean_accuracy_value

            if self.hide_meta:
                
                saver = session.graph.get_collection('savers')[0]
            
                saver.save(
                
                    session,
                    join( self.logdir, 'model.ckpt' ),
                    global_step = global_step,
                    latest_filename = None,
                    write_meta_graph = False, # dump weights only
                    write_state = True, # lookup file for the current checkpoint que saved on disk given max to keep
                    strip_default_attrs = False
                )
        else:
        
            current_eval_acc = mean_accuracy_value
            
            self.save_best_checkpoint(session, global_step)
                
class DetectorSessionRunHook(_SessionRunHOOK):
    
    def __init__(self, 
                
           iou_tensor,
           iou_update_op,
           batch_true_pos_tensor, 
           batch_false_pos_tensor,
           batch_false_neg_tensor,
           
           loss_op,
           reg_loss_op, 
           total_loss_op,
           
           global_step_op,
           logdir,
           tolerance_count,
           train_flag,
           hide_meta

    ):
        
        super().__init__(loss_op, reg_loss_op, total_loss_op, global_step_op, logdir, tolerance_count, train_flag, hide_meta)
    
        self.best_metrics_buffer = join( join(self.logdir, BEST_CKPT_DIR_NAME), BEST_METRICS_BUFFER_NAME )

        if isfile( self.best_metrics_buffer ):
            
            global best_train_mean_iou
            global best_train_f1_score
            global best_eval_mean_iou
            global best_eval_f1_score
            
            with open( self.best_metrics_buffer, 'r' ) as handle:
            
                dict = load( handle )
                
                best_train_mean_iou = float( dict[ BEST_TRAIN_IOU_KEY ] ) 
                best_train_f1_score = float( dict[ BEST_TRAIN_F1_SCORE_KEY ] )
                best_eval_mean_iou = float( dict[ BEST_EVAL_IOU_KEY ] )
                best_eval_f1_score = float( dict[ BEST_EVAL_F1_SCORE_KEY ] )
    
        self.iou_tensor = iou_tensor
        self.iou_update_op = iou_update_op
    
        self.true_pos_tensor = batch_true_pos_tensor
        self.false_pos_tensor = batch_false_pos_tensor
        self.false_neg_tensor = batch_false_neg_tensor
            
        self.total_true_pos = 0
        self.total_false_pos = 0
        self.total_false_neg = 0

        self.smallest_floating_point_number = float_info.min
        
        
    def save_best_checkpoint(self, session, global_step):
     
        global current_train_mean_iou
        global current_train_precision
        global current_train_recall
        global current_train_f1_score
                
        global current_eval_mean_iou
        global current_eval_precision
        global current_eval_recall
        global current_eval_f1_score
                
        global best_train_mean_iou
        global best_train_precision
        global best_train_recall
        global best_train_f1_score
                
        global best_eval_mean_iou
        global best_eval_precision
        global best_eval_recall
        global best_eval_f1_score

        '''
        
            TODO:
            
                mimic classifier platteau detection policy for object detector
        
        '''

        if (
        
            ( current_eval_f1_score > best_eval_f1_score ) or 
            ( (best_eval_f1_score == 1.0) and ( current_train_f1_score > best_train_f1_score ) )
        
        ):
            
            print('Found a better set of weights for the model ...')

            best_ckpt_dir = join(self.logdir, BEST_CKPT_DIR_NAME)
            if not isdir(best_ckpt_dir): mkdir(best_ckpt_dir)
           
            self.clear_old_ckpts(best_ckpt_dir)

            saver = session.graph.get_collection('savers')[0]
            
            saver.save(
                
                session,
                join( best_ckpt_dir, 'model.ckpt' ),
                global_step = global_step,
                latest_filename = None,
                write_meta_graph = not self.hide_meta, # dump weights only
                write_state = True, # lookup file for the current checkpoint que saved on disk given max to keep
                strip_default_attrs = False
            )

            best_train_mean_iou = current_train_mean_iou
            best_train_precision = current_train_precision
            best_train_recall = current_train_recall
            best_train_f1_score = current_train_f1_score
        
            best_eval_mean_iou = current_eval_mean_iou
            best_eval_precision = current_eval_precision
            best_eval_recall = current_eval_recall
            best_eval_f1_score = current_eval_f1_score
         
            TensorDyve.tensor_dyve.best_train_iou = best_train_mean_iou
            TensorDyve.tensor_dyve.best_train_f1_score = best_train_f1_score
            TensorDyve.tensor_dyve.best_eval_iou = best_eval_mean_iou
            TensorDyve.tensor_dyve.best_eval_f1_score = best_eval_f1_score
         
            with open(self.best_metrics_buffer, 'w') as handle:
                
                dict = {
                            
                            BEST_TRAIN_IOU_KEY: str( best_train_mean_iou ),
                            BEST_TRAIN_F1_SCORE_KEY: str( best_train_f1_score ),
                            BEST_EVAL_IOU_KEY: str( best_eval_mean_iou ),
                            BEST_EVAL_F1_SCORE_KEY: str( best_eval_f1_score )
                            
                }
                
                dump( dict, handle, indent = 5 )
         
    def before_run(self, run_context):
        
        return tf.train.SessionRunArgs( 
                                       
            [
             
                self.iou_update_op, # forcing metric_update_op to be executed after each batch in call to estimator.train
                self.loss_op,
                self.reg_loss_op,
                self.total_loss_op,
                self.true_pos_tensor,
                self.false_pos_tensor,
                self.false_neg_tensor,
                self.global_step_op,
                self.learning_rate_tensor,
                tf.train.get_global_step()
                                        
            ]
                                        
        )
     
    def after_run(self, run_context, run_values):
        
        ( batch_mean_iou, batch_loss, batch_reg_loss, batch_total_loss, batch_true_pos, 
          batch_false_pos, batch_false_neg, global_step, learning_rate, global_step ) = run_values.results
        
        precision = batch_true_pos / ( batch_true_pos + batch_false_pos + self.smallest_floating_point_number )
        recall = batch_true_pos / ( batch_true_pos + batch_false_neg + self.smallest_floating_point_number )
        
        if self.train_flag: # no need to log learning rate if evaluation is going on
            
            summary = tf.Summary()
            summary.value.add( tag = 'LearningRate', simple_value = learning_rate )
        
            self.summary_writer.add_summary( summary, global_step )
            self.summary_writer.flush()
        
        print(
                
                'Mean IOU for this session so far: ', batch_mean_iou, # this is a progressive mean, not a per batch stat
                  
                '\nBatch metrics for global step ', global_step, ':',
                '\nPrecision -->', precision, 
                '\nRecall -->', recall,
                '\nF1 Score -->', 2 * ( ( precision * recall ) / ( precision + recall + self.smallest_floating_point_number ) )
        )
        
        self.total_true_pos += batch_true_pos
        self.total_false_pos += batch_false_pos
        self.total_false_neg += batch_false_neg
        
        self.cumulative_loss += batch_loss
        self.cumulative_reg_loss += batch_reg_loss
        self.cumulative_total_loss += batch_total_loss
        self.seen_batches += 1
        
    def end(self, session):

        global current_train_mean_iou
        global current_train_precision
        global current_train_recall
        global current_train_f1_score
                
        global current_eval_mean_iou
        global current_eval_precision
        global current_eval_recall
        global current_eval_f1_score
                
        global best_train_mean_iou
        global best_train_precision
        global best_train_recall
        global best_train_f1_score
                
        global best_eval_mean_iou
        global best_eval_precision
        global best_eval_recall
        global best_train_f1_score

        mean_iou_value, global_step = session.run( [ self.iou_tensor, self.global_step_op ] ) # during the session the state of the mean_iou metric is update with the update op, at the end, given the state, evaluate the metric tensor and get the mean_iou for that particlar session
        loss_mean_value = self.cumulative_loss / self.seen_batches 
        reg_loss_mean_value = self.cumulative_reg_loss / self.seen_batches
        total_loss_mean_value = self.cumulative_total_loss / self.seen_batches
               
        precision = self.total_true_pos  /  ( self.total_true_pos + self.total_false_pos + self.smallest_floating_point_number )
        recall    = self.total_true_pos  /  ( self.total_true_pos + self.total_false_neg + self.smallest_floating_point_number )
        f1_score  = 2 * ( ( precision * recall ) / ( precision + recall + self.smallest_floating_point_number ) )
        
        print('------------------------------------------------------------------------------')
        print('END of ', self.mode_string, 'session')
        
        print( 'Mean IOU: ', mean_iou_value )
        print( 'Mean Loss: ' , loss_mean_value )
        print( 'Precision: ', precision )
        print( 'Recall: ',  recall )
        print( 'F1 Score: ', f1_score)
        
        print('------------------------------------------------------------------------------')
        
        summary = tf.Summary()
        
        summary.value.add( tag = 'Detector Metrics/Mean IOU', simple_value = mean_iou_value )
        summary.value.add( tag = 'Loss/Standalone', simple_value = loss_mean_value )
        summary.value.add( tag = 'Loss/Regularization', simple_value = reg_loss_mean_value )
        summary.value.add( tag = 'Loss/Total', simple_value = total_loss_mean_value )
        
        summary.value.add( tag = 'Detector Metrics/Precision', simple_value = precision)
        summary.value.add( tag = 'Detector Metrics/Recall', simple_value = recall )
        summary.value.add( tag = 'Detector Metrics/F1 Score', simple_value = f1_score )
        
        self.summary_writer.add_summary( summary, global_step )
        self.summary_writer.flush()
        
        if self.train_flag:
        
            current_train_mean_iou = mean_iou_value
            current_train_precision = precision
            current_train_recall = recall
            current_train_f1_score = f1_score

            
            if self.hide_meta:
                
                saver = session.graph.get_collection('savers')[0]
            
                saver.save(
                
                    session,
                    join( self.logdir, 'model.ckpt' ),
                    global_step = global_step,
                    latest_filename = None,
                    write_meta_graph = False,
                    write_state = True,
                    strip_default_attrs = False
                )    
                

        else:
            
            current_eval_mean_iou = mean_iou_value
            current_eval_precision = precision
            current_eval_recall = recall
            current_eval_f1_score = f1_score
            
            self.save_best_checkpoint(session, global_step)