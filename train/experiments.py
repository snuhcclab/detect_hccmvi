# SNU HCC LAB Junseo Kang
# Define Experiments to Run
# Not Called from Terminal
import os.path
import tensorflow as tf
from models.models import *
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.losses import BinaryCrossentropy, Loss
from dataclasses import dataclass
import pickle

HBP3D_SIZE = 224
DATA_HOME = 'data'


#
# Read Data from TFRecord File (8:1:1)
#
def decode(record_bytes, img_shape, multimodal=False):
    schema = {"image": tf.io.FixedLenFeature([], dtype=tf.string),
              "label": tf.io.FixedLenFeature([], dtype=tf.string)}
    schema_multimodal = {"image": tf.io.FixedLenFeature([], dtype=tf.string),
                         'tableur': tf.io.FixedLenFeature([], dtype=tf.string),
                         "label": tf.io.FixedLenFeature([], dtype=tf.string)}
    example = tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        schema if not multimodal else schema_multimodal
    )

    image = tf.reshape(tf.io.decode_raw(example["image"], tf.float64), img_shape)
    label = tf.io.decode_raw(example["label"], tf.int64)

    if not multimodal:
        return image, label

    else:
        tableur = tf.reshape(tf.io.decode_raw(example["tableur"], tf.float64), (4,))
        return {'image': image, 'tableur': tableur}, label


def decode_to_generator(dataset_name, scheme='original'):
    with open(os.path.join(DATA_HOME, 'temp_{0}'.format(dataset_name), '{0}.bin'.format(dataset_name)), 'rb') as pk:
        decoded = pickle.load(pk)

    def gen():
        if scheme == 'original':
            for ex in decoded:
                yield ex['image'], ex['label']
        elif scheme == 'multimodal':
            for ex in decoded:
                yield {'image': ex['image'],
                       'tableur': ex['tableur']}, ex['label']
    return gen


def read_dataset(dataset_name: str, batch_size=8, dataset_size=HBP3D_SIZE,
                 frac: tuple = (0.6, 0.2, 0.2), multimodal: bool = False, concat: bool = False):
    if concat:
        output_shape = ({'image': (96, 192, 12), 'tableur': (8)}, 1)
    elif multimodal:
        output_shape = ({'image': (96, 96, 12), 'tableur': (4)}, 1)
    else:
        output_shape = ((96, 96, 12), 1)

    dataset = tf.data.Dataset.from_generator(
        decode_to_generator(dataset_name, scheme='multimodal' if multimodal else 'original'),
        output_types=({'image': tf.float64, 'tableur': tf.float64}, tf.float64)
        if multimodal else (tf.float64, tf.float64),
        output_shapes=output_shape
    ).shuffle(batch_size * 128)

    dataset = dataset.batch(batch_size)
    cardinality = dataset_size / batch_size
    train_size, val_size, test_size = [int(f * cardinality) for f in frac]

    train_set = dataset.take(train_size)
    test_set = dataset.skip(train_size)
    val_set = test_set.skip(val_size)
    test_set = test_set.take(test_size)

    assert len(list(iter(test_set))) != 0

    return train_set, val_set, test_set


@dataclass
class Experiments:
    # 실험 이름, 모델, 데이터(훈련/밸리/텟), 로스, 옵티마이저, LR, 에포크, 배치 사이즈|
    # 텐서보드 여부, 얼리스타핑 여부, 얼리스타핑 에포크
    name: str
    model: Model
    train_data: tf.data.Dataset
    val_data: tf.data.Dataset
    test_data: tf.data.Dataset
    loss: Loss
    optimizer: str
    learning_rate: float
    epoch: int
    batch_size: int

    tensorboard: bool = False
    early_stopping: bool = False
    es_epoch: int = 16


hbp3d = read_dataset('hbp3d')
hbp3d_multimodal = read_dataset('hbp3d_multimodal', multimodal=True)
hbp3d_padding = read_dataset('hbp3d_padding')
hbp3d_padding_multimodal = read_dataset('hbp3d_padding_multimodal', multimodal=True)
concat_wjl = read_dataset('concat3d_wjl', multimodal=True, concat=True)

Res3D = Experiments(
    name='Res3D',
    model=resnet50_fine_tuning_multichannel((96, 96, 12), tune=36),
    train_data=hbp3d[0], val_data=hbp3d[1], test_data=hbp3d[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.0000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

Res3DMM = Experiments(
    name='Res3DMM',
    model=resnet50_fine_tuning_multichannel_multimodal_wjl_concat((96, 192, 12)),
    train_data=hbp3d_multimodal[0], test_data=hbp3d_multimodal[1], val_data=hbp3d_multimodal[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

Res3DPAD = Experiments(
    name='Res3DPAD',
    model=resnet50_fine_tuning_multichannel((96, 96, 12)),
    train_data=hbp3d_padding[0], test_data=hbp3d_padding[1], val_data=hbp3d_padding[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

Res3DMMPAD = Experiments(
    name='Res3DMMPAD',
    model=resnet50_fine_tuning_multichannel_multimodal((96, 96, 12), tune=118),
    train_data=hbp3d_padding_multimodal[0], test_data=hbp3d_padding_multimodal[1], val_data=hbp3d_padding_multimodal[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

VGG163D = Experiments(
    name='VGG163D',
    model=vgg16_multichannel((96, 96, 12)),
    train_data=hbp3d[0], val_data=hbp3d[1], test_data=hbp3d[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.0000000005,
    epoch=12,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

VGG163DMM = Experiments(
    name='VGG163DMM',
    model=vgg16_multichannel_multimodal((96, 96, 12)),
    train_data=hbp3d_multimodal[0], test_data=hbp3d_multimodal[1], val_data=hbp3d_multimodal[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=12,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

VGG163DPAD = Experiments(
    name='VGG163DPAD',
    model=vgg16_multichannel((96, 96, 12)),
    train_data=hbp3d_padding[0], test_data=hbp3d_padding[1], val_data=hbp3d_padding[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.00000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

VGG163DMMPAD = Experiments(
    name='VGG163DMMPAD',
    model=vgg16_multichannel_multimodal((96, 96, 12)),
    train_data=hbp3d_padding_multimodal[0], test_data=hbp3d_padding_multimodal[1], val_data=hbp3d_padding_multimodal[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

Den3D = Experiments(
    name='Den3D',
    model=densenet121_fine_tuning_multichannel((96, 96, 12)),
    train_data=hbp3d[0], test_data=hbp3d[1], val_data=hbp3d[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

Den3DMM = Experiments(
    name='EX_MCResMM',
    model=resnet50_fine_tuning_multichannel_multimodal((96, 96, 12)),
    train_data=hbp3d_multimodal[0], test_data=hbp3d_multimodal[1], val_data=hbp3d_multimodal[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)

CONRES3DMM = Experiments(
    name='EX_CONRES3DMM',
    model=resnet50_fine_tuning_multichannel_multimodal_wjl_concat((96, 192, 12)),
    train_data=concat_wjl[0], test_data=concat_wjl[1], val_data=concat_wjl[2],
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    learning_rate=0.000000005,
    epoch=128,
    batch_size=8,
    tensorboard=True,
    early_stopping=False
)
