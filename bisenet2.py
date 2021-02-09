import keras
from keras import backend as K
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Dropout, multiply, Dot, Concatenate,Add, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from myresnet import resnet_graph, load_weights

from keras.layers.core import Lambda
from keras.backend import tf as ktf
from keras.activations import softmax

def softMaxAxis3(x):
    return softmax(x,axis=3)


#-----------------


def conv_bn_act(inputs, n_filters=64, kernel=(3, 3), strides=1, activation='relu'):
    
    conv = Conv2D(n_filters, kernel_size= kernel, strides = strides, data_format='channels_last', padding='same')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    
    return conv


def conv_act(inputs, n_filters, kernel = (1,1), activation = 'relu', pooling = False):
    if pooling:
        conv = AveragePooling2D(pool_size=(1, 1), padding='same', data_format='channels_last')(inputs)
        conv = Conv2D(n_filters, kernel_size= kernel, strides=1)(conv)
        conv = Activation(activation)(conv)
    else:
        conv = Conv2D(n_filters, kernel_size= kernel, strides=1)(inputs)
        conv = Activation(activation)(conv)
    
    
    return conv


def CP_ARM(layer_13, layer_14):
    MyGlobalAveragePooling2D = Lambda(lambda t4d: K.mean(t4d, axis=(1,2), keepdims=True), name='GlobalAverage2D')
    # Combine the up-sampled output feature of Global avg pooling and Xception features
    tail = MyGlobalAveragePooling2D(layer_14)
    
    # ARM
    ARM_13 = ARM(layer_13, 1024)
    ARM_14 = ARM(layer_14, 2048)
    ARM_14 = multiply([tail, ARM_14])
    
    #layer_13 = UpSampling2D(size=2, data_format='channels_last')(ARM_13)
    layer_13 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (ARM_13)
    #layer_14 = UpSampling2D(size=4, data_format='channels_last')(ARM_14)
    layer_14 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (ARM_14)
    layer_14 = conv_bn_act(layer_14, 64, (3, 3))
    layer_14 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (layer_14)

    context_features = Concatenate(axis=-1)([layer_14, layer_13])
    
    return context_features

def ARM(inputs, n_filters):
    
    # ARM (Attention Refinement Module)
    # Refines features at each stage of the Context path
    # Negligible computation cost
    arm = AveragePooling2D(pool_size=(1, 1), padding='same', data_format='channels_last')(inputs)
    arm = conv_bn_act(arm, n_filters, (1, 1), activation='sigmoid')
    arm = multiply([inputs, arm])
    
    return arm


def FFM(input_sp, input_cp, n_classes):
    
    # FFM (Feature Fusion Module)
    # used to fuse features from the SP & CP
    # because SP encodes low-level and CP high-level features
    ffm = Concatenate(axis=-1)([input_sp, input_cp])
    conv = conv_bn_act(ffm, n_classes, (3, 3), strides= 2)
    
    conv_1 = conv_act(conv, n_classes, (1,1), pooling= True)
    conv_1 = conv_act(conv_1, n_classes, (1,1)) #'sigmoid'
    
    ffm1 = multiply([conv, conv_1])
    ffm1 = Add()([conv, ffm1])
    return ffm1



def bisenet(input_shape=None, num_classes=2, num_regress=0, act_regress=None, trainable_top = True, class_suffix = ''):

    # Context_path (Xception backbone and Attetion Refinement Module(ARM))
    #Xception_model = Xception(weights='imagenet',input_shape= (224,224,3), include_top=False)
    inputs, layer_14a, layer_13, _ = resnet_graph(input_shape, train_bn = True)
    # Context path & ARM
    CP_ARM0 = CP_ARM(layer_13, layer_14a)

    # Model (Input & Preprocession)
    #x = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(inputs)
    #x = Lambda(lambda image: preprocess_input(image))(x)

    # Spatial Path (conv_bn_act with strides = 2 )
    SP = conv_bn_act(inputs, 64, strides=2)
    SP = conv_bn_act(SP, 128, strides=2)
    SP = conv_bn_act(SP, 256, strides=2)

    # Feature Fusion Module(FFM)
    FFM0 = FFM(SP, CP_ARM0, 64) #num_classes

    # Upsampling the ouput to normal size
    layer_14 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (FFM0)
    layer_14 = conv_bn_act(layer_14, 64, (3, 3))
    layer_14 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (layer_14)
    layer_14 = conv_bn_act(layer_14, 64, (3, 3))
    layer_14 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (layer_14)
    layer_14 = conv_bn_act(layer_14, 64, (3, 3))
    layer_14 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (layer_14)
    next = Conv2D(num_classes, (1, 1), padding='same', activation=softMaxAxis3, name='classification' + class_suffix)(layer_14)
    if num_regress > 0:
        if type(act_regress) is list:
            lst = [next]
            for act in act_regress:
                regress = Conv2D(num_regress // len(act_regress), (1, 1), padding='same', activation=act)(layer_14)
                lst.append(regress)
        else:
            regress = Conv2D(num_regress, (1, 1), padding='same', activation=act_regress, name='regression')(layer_14)
            lst = [next, regress]
        #with tf.device("/cpu:0"):
        next = Concatenate(axis=-1)(lst)
    #output = UpSampling2D(size=(16,16), data_format='channels_last')(FFM0)
    
    model = Model(inputs = inputs, output = next)#[output, layer_13, layer_14])
    #load_weights(model, 'foto/Mask_RCNN-master/mask_rcnn_coco.h5')
    for layer in model.layers[:]:
        layer.trainable = trainable_top
        if layer.name == 'res5c_out':
            break
    return model

#print(bisnet.summary())

# We can visualize if our model was properly configure here
#from keras.utils import plot_model
#plot_model(bisnet, to_file='model.png')
