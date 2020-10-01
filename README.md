#TensorDyve is used for training, evaluating and exporting a user defined machine learning model intended for classification or YOLO object detection. Cab be used to train a classifier from crops out of annotated images.

#Inside a virtual enviroment of your choosing bound to a python3.x interpreter, run the command
#'pip install git+http://192.168.31.32:8080/tfs/ProjectsCollection/Dyve.MachineLearning/_git/DyveTFTraining'

#Within the virtual enviroment run 'tensor_dyve -h' for details on the command line arguments

#Usage example:

#*For training a detector from an image dataset you would use:*

#*tensor_dyve TRAIN --task DTT --config '/path/to/config/samples' --dump 'path/to/checkpoints/and/logging/dir' --img_dataset 'path/to/root/image/dataset' --color '0 for grayscale 1 for BGR' --extension '.ext for images' --percent 'train set percentage' --threshold '0 for now, will fix memory input pipe soon'*

#*For training a detector from an already existing tfr dataset you would use:*

#*tensor_dyve TRAIN --task DTT --config '/path/to/config/samples' --dump 'path/to/checkpoints/and/logging/dir' --tfr-dataset 'path/to/tfr/dataset/root/dir' *


#For predictions you would use:

#*tensor_dyve PREDICT --task DTT --config '/path/to/config/samples' --dump 'path/to/checkpoints/and/logging/dir' --pred_src_dir 'path/to/root/dir/for/images/to/make/predictions/on' --pred_dst_dir 'path/to/root/dir/where/predictions/go' --extension '.ext for images'*

#For a classifier just use CLS instead of DTT


#For exporting a trained model ready for inference just use:

#*tensor_dyve EXPORT --checkpoints_dir 'path/to/checkpoints/dir' --export_model_dir 'destination/directory/for/infernece/model/proto/file' --export_model_file_name 'proto file name'*