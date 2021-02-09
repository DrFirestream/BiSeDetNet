from keras_applications.resnet_common import ResNet, stack2
from keras_applications.resnet_v2 import ResNet101V2 as ResNet101o
from keras.layers import Convolution2D, Conv2DTranspose, concatenate, Dropout, Activation, BatchNormalization
from keras import layers as KL
import keras
from keras.activations import softmax
def softMaxAxis3(x):
    return softmax(x,axis=3)
def ResNet101(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4,stride1=1, name='conv3')
        x = stack2(x, 256, 23,stride1=1, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, True, True, 'resnet101v2',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, trainable_block = True):
    # first layer
    x = Convolution2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", trainable=trainable_block)(input_tensor)
    if batchnorm:
        x = BatchNormalization(trainable=trainable_block)(x)
    x = Activation("relu")(x)
    # second layer
    x = Convolution2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                padding="same", trainable=trainable_block)(x)
    if batchnorm:
        x = BatchNormalization(trainable=trainable_block)(x)
    x = Activation("relu")(x)
    return x
def BaseResNetUGraph(num_classes,inpt, outpt, c2, c1, nfilters = 128):
    x = outpt
#x = Convolution2D(64, (3, 3), padding='same', activation='relu')(x)

    u8 = Conv2DTranspose(nfilters*2, (3, 3), strides=(2, 2), padding='same', trainable=True, name = 'ubegin') (x)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.05, trainable=True)(u8)
    c8 = conv2d_block(u8, n_filters=nfilters*2, kernel_size=3, trainable_block=True)
    if not c1 is None:
        u9 = Conv2DTranspose(nfilters, (3, 3), strides=(2, 2), padding='same', trainable=True) (c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(0.05, trainable=True)(u9)
        c9 = conv2d_block(u9, n_filters=nfilters, kernel_size=3, trainable_block=True)
        x = c9
    else:
        x = c8
    return x
def BaseResNetU(num_classes,inpt, outpt, c2, c1, nfilters = 128):
    x = BaseResNetUGraph(num_classes, inpt, outpt, c2, c1, nfilters)
    x = Convolution2D(num_classes, (1, 1), padding='same', activation=softMaxAxis3, name='classification')(x)
    model_final = keras.models.Model(input = inpt, output = x)
    return model_final
def ResNet101u(config, trainable = False):
    model=ResNet101o(include_top=False, weights='imagenet', input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    for layer in model.layers[:]:
        layer.trainable = trainable
#    return BaseResNetU(config.NUM_CLASSES, model.input, model.output, model.get_layer('conv4_block23_out').output, model.get_layer('conv3_block4_out').output)
    return BaseResNetU(config.NUM_CLASSES, model.input, model.output, model.get_layer('conv4_block23_1_relu').output, model.get_layer('conv3_block4_1_relu').output)

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcumt
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(image_shape, architecture='resnet101', stage5=True, train_bn=False, strides=(2,2)):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    input_image = KL.Input(
            shape=[None, None, 3] if image_shape is None else [image_shape[0], image_shape[1], image_shape[2]], name="input_image")
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=strides, train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=strides, train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        return input_image, C5, C4, C3#[C1, C2, C3, C4, C5]
    else:
        C5 = None
        return input_image, C4, C3, None#[C1, C2, C3, C4, C5]


def load_weights(keras_model, filepath, by_name=True, exclude=None):
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    """
    import h5py
    # Conditional import to support versions of Keras before 2.2
    # TODO: remove in about 6 months (end of 2018)
    try:
        from keras.engine import saving
    except ImportError:
        # Keras before 2.2 used the 'topology' namespace.
        from keras.engine import topology as saving

    if exclude:
        by_name = True

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    #keras_model = self.keras_model
    layers = keras_model.layers
    #layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
    #   else keras_model.layers

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)

    if by_name:
        saving.load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=True)
    else:
        saving.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()

    # Update the log directory

def ResNet101au(config, stage5 = True):
    inpt, outpt, c2, c1 = resnet_graph(image_shape = (config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3), stage5=stage5)
    return BaseResNetU(config.NUM_CLASSES, inpt, outpt, c2, c1)

def ResNet101us(shape, num_classes, stage5 = True):
    inpt, outpt, c2, c1 = resnet_graph(image_shape = shape, stage5=stage5, train_bn = True)
    x = BaseResNetUGraph(num_classes, inpt, outpt, c2, c1)
    x = Convolution2D(num_classes, (1, 1), padding='same', activation='sigmoid', name='categorization')(x)
    return keras.models.Model(input = inpt, output = x)
