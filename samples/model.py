import tensorflow as tf

from imgaug.augmenters import Sometimes, Affine, Fliplr, Flipud
from imgaug.augmenters import GaussianBlur, Multiply, ContrastNormalization, Sequential, AdditiveGaussianNoise



'''

    In this file there should be one variable called AUG_PIPE and one function called model_function. The variable should contain the augmentation pipe
    object defined using the 'imgaug' API and the function should be the user defined computational graph using tensorflow, for convenience, the tf.layers API.
     
    The function signature should be:
    
        def model_function(images, train_flag):
        
            ------
            
                model goes in here
                
            ------
        
        return user_defined_output_op, user_defined_optimizer_op
    
    images     -> batch of images
    train_flag -> whether the estimator API is using the graph in a train context or not, can be used for building the dropout nodes properly, see ' tf.layers.dropout ' 

    NOTE: 
    
        The user defined computation graph should only be the feedforward version, the training specifics are handled by the utility, loss and other
        metrics are added later on, as well as any tensorboard logging ops. In case of a yolo model, the yolo layer at the end will be appended. All the relevant information about the model
        and training will be appended to the initial 'config.json', so that it can be referenced at a later point, thus the training is bound to the
        dataset used. At the moment you can only train a yolo object detector, for which you need to return the op preceding the yolo layer, and standalone mlp or convnet + mlp classifiers, 
        for which you need to return the op preceding the logits op, since it can infer the number of classes, thus being able to append the last layer.
        
        The user can also define the optimizer op.

    
    https://imgaug.readthedocs.io/en/latest: imgaug API for dataset augmentations
    
    https://www.tensorflow.org/api_guides/python/train: tf.train API for building optimizers and other utilities like learning rate decay policies 
    https://www.tensorflow.org/api_docs/python/tf/layers: tf.layers API for building the model, other high level apis like SLIM can be used, thus the SLIM defined models like the popular RESNET or MOBILENETS is customizable and can be used out of the box
    
    !!!! MAKE SURE YOU SPECIFY THE CLASS_WEIGHTS PARAM PROPERLY IN 'config.json' when training a YOLO DTT i.e. num classes has to match the number of class weights

    If doing quantized training with toco compatibility there are a few constraints, namely:

        1. Minimum Tensorflow version has to be 1.13
        2. The current supported non-linearity elements are relu and relu6
        3. The convolution op instantiation has to have the non-linearity element set via the 'activation' argument
        4. The batch normalization op instantiation has to have the 'fused' argument set to false

'''

AUG_PIPE = Sequential( 


                [
                    Sometimes( 0.33, Affine( rotate = (-5, 5))),
                    Sometimes( 0.33, Affine( scale = {"x": (0.85, 1.15), "y": (0.85, 1.15) } ) ),
                    Sometimes( 0.33, Affine( shear = (-8, 8))),
                    Sometimes( 0.33, Multiply((0.9, 1.1)) ),
                    Sometimes( 0.33, AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255))),
                    Sometimes( 0.33, ContrastNormalization((0.9, 1.1)))

                ],

                random_order = True

)

AUG_PIPE = None # this is how you skip doing any augmentations, otherwise just have AUG_PIPE contain a imgaug Sequential instance

def model_function(images, train_flag):
    
    '''
    
        The 'name' argument for each of the op building functions can be whatever you want except 'input', 'prediction', 'loss', 'accuracy' 
        which are reserved for convenience. The names are the ones that will show up in tensorboard when viewing the op graph structure and can also be used to reference the input or output tensors of these ops.
    
        Important note:
        
            User has to define a variable or constant or learning_rate_decay_policy with the name 'learning_rate', even if the learning rate is fixed,
            this is done in order to have tensorboard loging for the learning rate in general. Reason being, different types of optimizers have
            the intended private variable 'learning_rate' defined under different names. Also since it's an intended private variable,
            it's not intended for public use duh, it's name might change with different TF versions. Suprisingly, there is no geter-like way to
            fetch the learning rate, except for interogating the graph after construction, knowing it's name.
            
    '''
    
    decaying_learning_rate = tf.train.exponential_decay(
    
        learning_rate = 0.01,
        global_step = tf.train.get_global_step(),
        decay_steps = 1000,
        decay_rate = 0.9,
        staircase=False,
        name='learning_rate'
    )

    optimizer = tf.train.AdamOptimizer(

        learning_rate = decaying_learning_rate,

        name='AdamOptimizer'
    )
    
    regularizer = tf.contrib.layers.l2_regularizer( scale = 0.0005 )

    cv2d_1 = tf.layers.conv2d(

        inputs=images, 
        filters=16,
        bias_regularizer = regularizer,
        kernel_regularizer = regularizer, 
        kernel_size=3, 
        padding='same', 
        strides=1, 
        activation=None,
        
        #trainable = train_flag, 
        
        name='cv2d_1'
    )

    bn_1 = tf.layers.batch_normalization(
    
        cv2d_1,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        
        beta_regularizer=None, # one can pass l2_regularizer to this also
        gamma_regularizer=None, # one can pass l2_regularizer to this also
       
        beta_constraint=None,
        gamma_constraint=None,
        
        training=train_flag,
        #trainable = train_flag,
        
        name='bn_1',
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        virtual_batch_size=None,
        adjustment=None
    )
    
    lrelu_1 = tf.nn.leaky_relu(features=bn_1, name='lrelu_1')
    mp_1 = tf.layers.max_pooling2d(inputs=lrelu_1, pool_size=2, strides=2, padding='same', name='mp_1')
    
    cv2d_2 = tf.layers.conv2d(
        inputs=mp_1, 
        filters=32,
        bias_regularizer = regularizer,
        kernel_regularizer = regularizer,  
        kernel_size=3, 
        padding='same', 
        strides=1, 
        activation=None, 
        
        #trainable = train_flag,
        
        name='cv2d_2'
    )
    
    bn_2 = tf.layers.batch_normalization(
    
        cv2d_2,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        
        beta_regularizer=None, # one can pass a regularizer to this also
        gamma_regularizer=None, # one can pass a regularizer to this also
       
        beta_constraint=None,
        gamma_constraint=None,
        
        training=train_flag,
        #trainable = train_flag,
        
        name='bn_2',
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        virtual_batch_size=None,
        adjustment=None
    )
    
    lrelu_2 = tf.nn.leaky_relu(features=bn_2, name='lrelu_2')
    mp_2 = tf.layers.max_pooling2d(inputs=lrelu_2, pool_size=2, strides=2, padding='same', name='mp_2')
    
    cv2d_3 = tf.layers.conv2d(

        inputs=mp_2,
        bias_regularizer = regularizer,
        kernel_regularizer = regularizer,  
        filters=24, 
        kernel_size=3, 
        padding='same', 
        strides=1, 
        activation=None,

        #trainable = train_flag, 
        
        name='cv2d_3'
    )

    bn_3 = tf.layers.batch_normalization(
    
        cv2d_3,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        
        beta_regularizer=None, # one can pass a regularizer to this also
        gamma_regularizer=None, # one can pass a regularizer to this also
       
        beta_constraint=None,
        gamma_constraint=None,

        training = train_flag,
        #trainable = train_flag,

        name='bn_3',
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        virtual_batch_size=None,
        adjustment=None
    )
    
    lrelu_3 = tf.nn.leaky_relu(features=bn_3, name='lrelu_3')
    mp_3 = tf.layers.max_pooling2d(inputs=lrelu_3, pool_size=2, strides=2, padding='same', name='mp_3')
    
    cv2d_4 = tf.layers.conv2d(

        inputs=mp_3,
        bias_regularizer = regularizer,
        kernel_regularizer = regularizer,  
        filters=48, 
        kernel_size=3, 
        padding='same', 
        strides=1, 
        activation=None, 
        
        #trainable = train_flag,
        
        name='cv2d_4'
    )

    bn_4 = tf.layers.batch_normalization(
    
        cv2d_4,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        
        beta_regularizer=None, # one can pass a regularizer to this also
        gamma_regularizer=None, # one can pass a regularizer to this also
       
        beta_constraint=None,
        gamma_constraint=None,
        
        training = train_flag,
        #trainable = train_flag,
        
        name='bn_4',
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        virtual_batch_size=None,
        adjustment=None
    )
    
    lrelu_4 = tf.nn.leaky_relu(features=bn_4, name='lrelu_4')
    mp_4 = tf.layers.max_pooling2d(inputs=lrelu_4, pool_size=2, strides=2, padding='same', name='mp_4')
    
    cv2d_5 = tf.layers.conv2d(

        inputs=mp_4,
        bias_regularizer = regularizer,
        kernel_regularizer = regularizer,  
        filters=64, 
        kernel_size=3, 
        padding='same', 
        strides=1, 
        activation=None,
        
        #trainable = train_flag, 
        
        name='cv2d_5'
    )
    
    bn_5 = tf.layers.batch_normalization(
    
        cv2d_5,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        
        beta_regularizer=None, # one can pass a regularizer to this also
        gamma_regularizer=None, # one can pass a regularizer to this also
       
        beta_constraint=None,
        gamma_constraint=None,
        
        training = train_flag,
        #trainable = train_flag,
        
        name='bn_5',
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        virtual_batch_size=None,
        adjustment=None
    )
    lrelu_5 = tf.nn.leaky_relu(features=bn_5, name='lrelu_5')
    mp_5 = tf.layers.max_pooling2d(inputs=lrelu_5, pool_size=2, strides=2, padding='same', name='mp_5')
    
    cv2d_6 = tf.layers.conv2d(

        inputs=mp_5,
        bias_regularizer = regularizer,
        kernel_regularizer = regularizer,  
        filters=128, 
        kernel_size=3, 
        padding='same', 
        strides=1, 
        activation=None, 
        
        #trainable = train_flag,
        
        name='cv2d_6'
    )

    bn_6 = tf.layers.batch_normalization(
    
        cv2d_6,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        
        beta_regularizer=None, # one can pass a regularizer to this also
        gamma_regularizer=None, # one can pass a regularizer to this also
       
        beta_constraint=None,
        gamma_constraint=None,
        
        training=train_flag,
        #trainable = train_flag,
        
        name='bn_6',
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused = None,
        virtual_batch_size=None,
        adjustment=None
    )

    lrelu_6 = tf.nn.leaky_relu(features=bn_6, name='lrelu_5')

    # in this case there will be a full connection between this flat feature vector and the output layer, any more hidden layers can be chained
    fully_connected_input = tf.layers.flatten(lrelu_6, name = 'fully_connected_input')
    
    return fully_connected_input, optimizer
    
