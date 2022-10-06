#
# read AP/HBP Images and patient data per each, fit each model, plot them
# python pipeline.py ap_dir hbp_dir data_dir temp_dir result_dir

import os
import shutil
import sys
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from numpy import ndarray
from itertools import product
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Dense, ReLU, Input, BatchNormalization, LeakyReLU, Flatten, Dropout
from keras.layers import GlobalAvgPool2D
from keras.layers import RandomFlip, RandomRotation, Rescaling, Resizing, concatenate
from keras.models import Sequential, Model, save_model, load_model
from keras.applications import MobileNetV2, DenseNet121, ResNet50, VGG16, VGG19
from keras.optimizers import Adam, RMSprop
from keras.losses import BinaryCrossentropy

CUT_TO_USE = 12
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 8
FRAC = (0.8, 0.2, 0.2)
EPOCH = 64


def process_dataset(patient_list, data_path, data_type='HBP'):
    patient_data = pd.read_excel(data_path).dropna()

    def make_generator():
        for patient in patient_list:
            try:
                patient_record = patient_data.loc[patient_data['patID'].map(int) == int(patient[-7:])]
            except ValueError:
                continue

            #
            # Read Tumor Cuts and Stack Images
            #
            tumor_cut_label = 'tumor(arterial)' if data_type == 'AP' else 'tumor(20min)'

            if not len(patient_record[tumor_cut_label]) == 0:
                tumor_cut = [int(x) for x in patient_record['tumor(20min)'].str.split(',').tolist()[0]]
            else:
                continue

            temp = []

            for i, cut in enumerate(os.listdir(patient)):
                if tumor_cut[0] <= i <= tumor_cut[1]:
                    img = Image.open(os.path.join(patient, cut)).convert('L')
                    img = np.array(img).reshape((96, 192, 1)) / 255 + 1e-5  # Preprocess
                    temp.append(img)

            if len(temp) > CUT_TO_USE:
                temp = temp[:CUT_TO_USE]
            elif len(temp) < CUT_TO_USE:
                temp = temp + [np.zeros(shape=(96, 192, 1)) + 1e-5 for _ in range(CUT_TO_USE - len(temp))]

            img_3d: ndarray = np.concatenate(temp, axis=2)

            #
            # Read Tableur Data
            #

            tableur = patient_record[['gender', 'age', 'cirrhosis', 'Tumor_size1']].values.tolist()[0]
            tumor_z_size = (tumor_cut[1] - tumor_cut[0]) / len(os.listdir(patient))
            tableur.append(tumor_z_size)

            #
            # Read Label
            #

            if data_type == 'AP':
                records_to_read = ['peritumoral enhancement(arterial)',
                                   'rim-like enhancement(arterial)']
            elif data_type == 'HBP':
                records_to_read = ['peritumoral hypointensity(20min)',
                                   'Irregular margin(20min)']
            else:
                continue

            label = patient_record[records_to_read].values.tolist()[0]
            neg_sig = ['0', '0m', 'x', '?']
            label = [0 if str(x) in neg_sig else 1 for x in label]
            label = tuple(label)

            def make_combination(ser, n):
                items = [(0, 1) for _ in range(n)]
                able = list(product(*items))
                able = sorted(able,
                              key=lambda x: x.count(1))
                idx = able.index(ser)
                return np.array([1e-5 if x != idx else 1 - 1e-5 for x in range(2 ** n)], float)

            label = make_combination(label, n=2)

            #
            # Check Machine and Tumor Size, Select
            #
            machine = patient_record[['Machine']].values.tolist()[0][0]

            if tableur[3] <= 2 or not machine.startswith('A'):
                continue

            #
            # Normalize Tableur data
            #

            tableur[0] = 1e-5 if tableur[0] == 'M' else 1 - 1e-5
            tableur[1] = tableur[1] / 110 + 1e-5  # Age
            tableur[2] = 1e-5 if tableur[2] == 0 else 1 - 1e-5
            tableur[3] = (tableur[3] - 1e-5) / 5
            tableur = np.array(tableur, float)

            #
            # yield packed data
            #
            yield {'image': img_3d,
                   'tableur': tableur,
                   'label': label}

    return make_generator()


def build_resnet():
    pass


def main(raw_path: dict, data_path, temp_dir, model_dir, plot_dir):
    # Check Temp dir, if not?
    # Read Images, Patient Data and Pickle it
    # raw_path['AP'], raw_path['HBP']
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        for data_type in ('AP', 'HBP'):
            print(data_type)

            patient_list = sum([
                [os.path.join(raw_path[data_type], x, y) for y in os.listdir(os.path.join(raw_path[data_type], x))]
                for x in ('0', '1')],
                [])
            temp = process_dataset(patient_list, data_path, data_type=data_type)
            os.mkdir(os.path.join(temp_dir, data_type))
            with open(os.path.join(temp_dir, data_type, 'pickled_data' + '.bin'), 'wb') as pk:
                pickle.dump(list(temp), pk)

    else:
        print("Temp Data Checked, Skipping..")

    # Check Model dir, if not?
    # Fit Neural Network
    #
    if not os.path.exists(model_dir):
        #
        # For Each Model
        #
        os.mkdir(model_dir)

        for data_type in ('AP', 'HBP'):
            #
            # Make Directory
            #
            os.mkdir(os.path.join(model_dir, data_type))
            os.mkdir(os.path.join(model_dir, data_type, 'log'))
            os.mkdir(os.path.join(model_dir, data_type, 'model'))

            #
            # Read Data, make tf.Data
            #
            with open(os.path.join(temp_dir, data_type, 'pickled_data.bin'), 'rb') as pk:
                loaded_data = pickle.load(pk)
            dataset_size = len(loaded_data)

            def make_generator(data):
                def gen():
                    for record in data:
                        yield {'image': record['image'],
                               'tableur': record['tableur']}, record['label']
                return gen

            dataset = tf.data.Dataset.from_generator(
                make_generator(loaded_data),
                output_types=({'image': tf.float64, 'tableur': tf.float64}, tf.float64),
                output_shapes=({'image': (96, 192, 12), 'tableur': 5}, 4)
            ).shuffle(BATCH_SIZE * 128)

            dataset = dataset.batch(BATCH_SIZE)
            cardinality = dataset_size / BATCH_SIZE

            train_size, val_size, test_size = [int(f * cardinality) for f in FRAC]

            train_set = dataset.take(train_size)
            dataset.skip(train_size)
            val_set = dataset.take(val_size)
            dataset.skip(val_size)
            test_set = dataset.take(test_size)

            assert len(list(iter(test_set))) != 0

            #
            # Build Model
            #
            augmentation_preprocessing = Sequential(
                [RandomFlip('horizontal'), RandomRotation(0.2), Rescaling(1. / 127.5, offset=-1),
                 Resizing(112, 224)])

            compress_layer = Conv2D(3, kernel_size=1, strides=1)

            base_model = ResNet50(input_shape=(112, 224, 3),
                                  include_top=False,
                                  weights='imagenet')

            base_model.trainable = True
            for layer in base_model.layers[:128]:
                layer.trainable = False

            ga_layer = GlobalAvgPool2D()

            prediction_layer = Dense(4, activation='softmax')

            inputs_ = Input(shape=(96, 192, 12))
            sub_inputs_ = Input(shape=(5,))

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

            #
            # Fit Model
            #

            callbacks = [TensorBoard(log_dir=os.path.join(model_dir, data_type, 'log'))]

            model.compile(
                optimizer=Adam(learning_rate=5e-6),
                loss=BinaryCrossentropy(),
                metrics=['categorical_accuracy']
            )

            history = model.fit(train_set,
                                validation_data=val_set,
                                epochs=EPOCH,
                                callbacks=callbacks)

            save_model(model, os.path.join(model_dir, data_type, 'model'))

            #
            # Save Model
            #

    else:
        print("Temp Data Checked, Skipping..")
    #
    # Fit Machine Learning
    #

    #
    # Plot Model
    #


if __name__ == '__main__':
    main(raw_path={
        'AP': sys.argv[1],
        'HBP': sys.argv[2]
    },
        data_path=sys.argv[3],
        temp_dir=sys.argv[4],
        model_dir=sys.argv[5],
        plot_dir=sys.argv[6]
    )
