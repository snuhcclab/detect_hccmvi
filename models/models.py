import numpy
from keras.layers import Conv2D, MaxPool2D, Dense, ReLU, Input, BatchNormalization, LeakyReLU, Flatten, Dropout
from keras.layers import GlobalAvgPool2D
from keras.layers import RandomFlip, RandomRotation, Rescaling, Resizing, concatenate
from keras.models import Sequential, Model
from keras.applications import MobileNetV2, DenseNet121, ResNet50, VGG16, VGG19
from keras.optimizers import Adam, RMSprop


def vanilla_cnn(input_shape):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    base_model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dropout(0.2),
        Dense(1)
    ])

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    outputs_ = base_model(x)

    model = Model(inputs_, outputs_)
    model.summary()
    return model


def vgg16(input_shape):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    base_model = VGG16(input_shape=input_shape,
                       include_top=False)
    base_model.trainable = True

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    x = base_model(x)
    x = ga_layer(x)
    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)

    return model


def vgg16_multichannel(input_shape):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    compress_layer_1 = Conv2D(12, kernel_size=1, strides=1)
    compress_layer_2 = Conv2D(6, kernel_size=1, strides=1)
    compress_layer_3 = Conv2D(3, kernel_size=1, strides=1)

    base_model = VGG16(input_shape=(*input_shape[:2], 3),
                       include_top=False)
    base_model.trainable = True

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    x = compress_layer_1(x)
    x = compress_layer_2(x)
    x = compress_layer_3(x)
    x = base_model(x)
    x = ga_layer(x)
    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)

    return model


def vgg16_multichannel_multimodal(input_shape):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    compress_layer_1 = Conv2D(12, kernel_size=1, strides=1)
    compress_layer_2 = Conv2D(6, kernel_size=1, strides=1)
    compress_layer_3 = Conv2D(3, kernel_size=1, strides=1)

    base_model = VGG16(input_shape=(*input_shape[:2], 3),
                       include_top=False)
    base_model.trainable = True

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)
    sub_inputs_ = Input(shape=(4,))

    x = augmentation_preprocessing(inputs_)
    x = compress_layer_1(x)
    x = compress_layer_2(x)
    x = compress_layer_3(x)
    x = base_model(x)
    x = ga_layer(x)

    x_s = Dense(8)(sub_inputs_)
    x = concatenate([x, x_s])

    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model({
        'image': inputs_,
        'tableur': sub_inputs_
    }, outputs_)
    model.summary()
    return model


def mobilenet_fine_tuning(input_shape, tune=100):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:tune]:
        layer.trainable = False

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    x = base_model(x)
    x = ga_layer(x)
    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)

    return model


def densenet121_fine_tuning(input_shape, tune=120):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    base_model = DenseNet121(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:tune]:
        layer.trainable = False

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    x = base_model(x)
    x = ga_layer(x)
    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)
    return model


def densenet121_fine_tuning_multichannel(input_shape, tune=120):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    compress_layer_1 = Conv2D(12, kernel_size=1, strides=1)
    compress_layer_2 = Conv2D(6, kernel_size=1, strides=1)
    compress_layer_3 = Conv2D(3, kernel_size=1, strides=1)

    base_model = DenseNet121(input_shape=(*input_shape[:2], 3),
                             include_top=False,
                             weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:tune]:
        layer.trainable = False

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    x = compress_layer_1(x)
    x = compress_layer_2(x)
    x = compress_layer_3(x)
    x = base_model(x)
    x = ga_layer(x)
    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)
    model.summary()
    return model


def resnet50_fine_tuning_multichannel(input_shape, tune=40):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    compress_layer_1 = Conv2D(12, kernel_size=1, strides=1)
    compress_layer_2 = Conv2D(6, kernel_size=1, strides=1)
    compress_layer_3 = Conv2D(3, kernel_size=1, strides=1)

    base_model = ResNet50(input_shape=(*input_shape[:2], 3),
                          include_top=False,
                          weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:tune]:
        layer.trainable = False

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)

    x = augmentation_preprocessing(inputs_)
    x = compress_layer_1(x)
    x = compress_layer_2(x)
    x = compress_layer_3(x)
    x = base_model(x)
    x = ga_layer(x)
    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model(inputs_, outputs_)
    model.summary()
    return model


def resnet50_fine_tuning_multichannel_multimodal(input_shape, tune=48):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1)])

    compress_layer_1 = Conv2D(12, kernel_size=1, strides=1)
    compress_layer_2 = Conv2D(6, kernel_size=1, strides=1)
    compress_layer_3 = Conv2D(3, kernel_size=1, strides=1)

    base_model = ResNet50(input_shape=(*input_shape[:2], 3),
                          include_top=False,
                          weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:tune]:
        layer.trainable = False

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(1)

    inputs_ = Input(shape=input_shape)
    sub_inputs_ = Input(shape=(4,))

    x = augmentation_preprocessing(inputs_)
    x = compress_layer_1(x)
    x = compress_layer_2(x)
    x = compress_layer_3(x)
    x = base_model(x)

    x_s = Dense(8)(sub_inputs_)

    x = ga_layer(x)

    x = concatenate([x, x_s])

    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model({
        'image': inputs_,
        'tableur': sub_inputs_
    }, outputs_)
    model.summary()
    return model


def resnet50_fine_tuning_multichannel_multimodal_wjl_concat(input_shape, tune=128):
    augmentation_preprocessing = Sequential(
        [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1),
         Resizing(112, 224)])

    compress_layer = Conv2D(3, kernel_size=1, strides=1)

    base_model = ResNet50(input_shape=(112, 224, 3),
                          include_top=False,
                          weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:tune]:
        layer.trainable = False

    ga_layer = GlobalAvgPool2D()

    prediction_layer = Dense(16, activation='sigmoid')

    inputs_ = Input(shape=input_shape)
    sub_inputs_ = Input(shape=(5, ))

    x = augmentation_preprocessing(inputs_)

    x = compress_layer(x)

    x = base_model(x)

    x_s = Dense(8)(sub_inputs_)

    x = ga_layer(x)

    x = concatenate([x, x_s])

    x = Dropout(0.2)(x)

    outputs_ = prediction_layer(x)

    model = Model({
        'image': inputs_,
        'tableur': sub_inputs_
    }, outputs_)
    model.summary()
    return model
