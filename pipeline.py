#
# read AP/HBP Images and patient data per each, fit each model, plot them
# python pipeline.py raw_dir data_dir temp_dir result_dir model

import os
import sys
import pickle
import warnings
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
from PIL import Image
from numpy import ndarray
from itertools import product
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Dense, ReLU, Input, BatchNormalization, LeakyReLU, Flatten, Dropout
from keras.layers import GlobalAvgPool2D
from keras.layers import RandomFlip, RandomRotation, Rescaling, Resizing, concatenate
from keras.models import Sequential, Model, save_model, load_model
from keras.applications import MobileNetV2, DenseNet121, ResNet50, VGG16, VGG19, EfficientNetV2B0
from keras.optimizers import Adam, RMSprop
from keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, auc, f1_score, confusion_matrix, recall_score
from xgboost import XGBClassifier

CUT_TO_USE = 12
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 4
FRAC = (0.8, 0.1, 0.1)
EPOCH = 96
warnings.filterwarnings('ignore')


def process_dataset(patient_list, data_path, input_shape, data_type='HBP'):
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
                    img = np.array(img).reshape((*input_shape, 1)) / 255 + 1e-5  # Preprocess
                    temp.append(img)

            if len(temp) > CUT_TO_USE:
                temp = temp[:CUT_TO_USE]
            elif len(temp) < CUT_TO_USE:
                temp = temp + [np.zeros(shape=(*input_shape, 1)) + 1e-5 for _ in range(CUT_TO_USE - len(temp))]

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
            # Append MVI Data
            #
            mvi = patient_record[['MVI']].values.tolist()[0][0]
            mvi = 1e-5 if mvi == 0.0 else 1 - 1e-5

            #
            # yield packed data
            #
            yield {'image': img_3d,
                   'tableur': tableur,
                   'label': label,
                   'mvi': mvi}

    return make_generator()


def fit_xgboost(patient_data):
    #
    # Pre-Process
    #
    patient_df: pd.DataFrame
    patient_df = pd.read_excel(patient_data)
    patient_df = patient_df[[
        'gender', 'age', 'cirrhosis', 'Tumor_size1', 'tumor(20min)',
        'peritumoral enhancement(arterial)', 'rim-like enhancement(arterial)',
        'peritumoral hypointensity(20min)', 'Irregular margin(20min)', 'MVI'
    ]].dropna()
    patient_df.columns = ['Gender', 'Age', 'Cirrhosis', 'Size', 'tumor(20min)',
                          'PEA', 'REA', 'RHH', 'IMH', 'MVI']

    def cut_to_size(record):
        if not len(record) == 0:
            temp = [int(x) for x in record.split(',')]
            return temp[1] - temp[0]
        else:
            return pd.NA

    patient_df['tumor(20min)'] = patient_df['tumor(20min)'].map(cut_to_size)

    def cut_to_exist(record):
        neg_sig = ['0', '0m', 'x', '?', '0.0']
        if str(record) in neg_sig:
            return 0
        else:
            return 1

    cols = ['PEA', 'REA', 'RHH', 'IMH', 'MVI']
    for col in cols:
        patient_df[col] = patient_df[col].map(cut_to_exist)
    patient_df['Age'] = pd.to_numeric(patient_df['Age'] / 110)
    patient_df['Size'] = patient_df['Size'] / 5
    patient_df['Gender'] = pd.to_numeric(patient_df['Gender'].map(lambda i: 1e-5 if i == 'M' else 1 - 1e-5))
    patient_df['Cirrhosis'] = pd.to_numeric(patient_df['Cirrhosis'])

    #
    # Fit Model
    #
    x = patient_df.drop(['MVI'], axis=1)
    y = patient_df['MVI']

    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=1956)
    model = XGBClassifier(n_estimators=100, n_jobs=-1, max_depth=None, random_state=1956,
                          objective='binary:logistic')

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    importance = model.feature_importances_

    return model


def make_generator(data):
    def gen():
        for record in data:
            yield {'image': record['image'],
                   'tableur': record['tableur']}, record['label']

    return gen


def main(raw_path: dict, data_path, temp_dir, model_dir, plot_dir, base, data_shape):
    # Check Temp dir, if not?
    # Read Images, Patient Data and Pickle it
    #
    input_shape: tuple
    if data_shape == '224':
        input_shape = (224, 224)
    else:
        input_shape = (96, 192)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        for data_type in ('AP', 'HBP'):
            patient_list = [os.path.join(raw_path[data_type], x) for x in os.listdir(raw_path[data_type])]
            temp = list(process_dataset(patient_list, data_path, data_type=data_type, input_shape=input_shape))
            dataset_size = len(temp)
            print('Dataset Size: ', str(dataset_size))
            train_size = int(dataset_size * (FRAC[0] + FRAC[1]))
            os.mkdir(os.path.join(temp_dir, data_type))
            with open(os.path.join(temp_dir, data_type, 'pickled_data_train' + '.bin'), 'wb') as pk:
                pickle.dump(temp, pk)
            with open(os.path.join(temp_dir, data_type, 'pickled_data_test' + '.bin'), 'wb') as pk:
                pickle.dump(temp, pk)

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
            with open(os.path.join(temp_dir, data_type, 'pickled_data_train.bin'), 'rb') as pk:
                loaded_data = pickle.load(pk)
            dataset_size = len(loaded_data)

            dataset = tf.data.Dataset.from_generator(
                make_generator(loaded_data),
                output_types=({'image': tf.float64, 'tableur': tf.float64}, tf.float64),
                output_shapes=({'image': (*input_shape, 12), 'tableur': 5}, 4)
            ).shuffle(BATCH_SIZE * 128)

            dataset = dataset.batch(BATCH_SIZE)
            cardinality = dataset_size / BATCH_SIZE

            print('card', cardinality)

            train_size, val_size = [int(f / (FRAC[0] + FRAC[1]) * cardinality) for f in FRAC[:2]]

            train_set = dataset.take(train_size)
            dataset.skip(train_size)
            val_set = dataset.take(val_size)
            dataset.skip(val_size)

            assert len(list(iter(val_set))) != 0

            #
            # Build Model
            #
            base_model: keras.Model
            if base == 'EFF':
                base_model = EfficientNetV2B0(input_shape=(*input_shape, 3),
                                              include_top=False,
                                              weights='imagenet')
            elif base == 'RES':
                base_model = ResNet50(input_shape=(*input_shape, 3),
                                      include_top=False,
                                      weights='imagenet')
            elif base == 'VGG':
                base = VGG16(input_shape=(*input_shape, 3),
                                   include_top=False,
                                   weights='imagenet')
            elif base == 'DEN':
                base_model = DenseNet121(input_shape=(*input_shape, 3),
                                         include_top=False,
                                         weights='imagenet')
                print('Densenet')
            else:
                print(base)
                exit()
            try:
                base_model.trainable = True
            except UnboundLocalError:
                print(base)
                exit()
            for layer in base_model.layers[:100]:
                layer.trainable = False

            augmentation_preprocessing = Sequential(
                [RandomFlip('horizontal'), RandomRotation(0.3)])

            compress_layer_1 = Conv2D(8, kernel_size=1, strides=1)
            compress_layer_2 = Conv2D(3, kernel_size=1, strides=1)

            ga_layer = GlobalAvgPool2D()

            prediction_layer = Dense(4, activation='softmax')

            inputs_ = Input(shape=(*input_shape, 12))
            sub_inputs_ = Input(shape=(5,))

            x = augmentation_preprocessing(inputs_)

            x = compress_layer_1(x)
            x = compress_layer_2(x)

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
                optimizer=Adam(learning_rate=5e-7),
                loss=BinaryCrossentropy(),
                metrics=['categorical_accuracy']
            )

            history = model.fit(train_set,
                                validation_data=val_set,
                                epochs=EPOCH,
                                callbacks=callbacks)

            save_model(model, os.path.join(model_dir, data_type, 'model'))
            # TODO: Plot Training History

        #
        # Fit ML Model
        #
        xgboost_model = fit_xgboost(
            data_path
        )
        with open(os.path.join(model_dir, 'model.xgb'), 'wb') as xgb:
            pickle.dump(xgboost_model, xgb)

    else:
        print("Model Data Checked, Skipping..")

    # Check Plot dir, if not?
    # Plot Model
    #
    if not os.path.exists(plot_dir):
        #
        # Make Plot dir
        #
        os.mkdir(plot_dir)
        print('Plotting')
        #
        # For Each Data Style
        #
        temp_pred = []
        temp_real = []

        for data_type in ('AP', 'HBP'):
            #
            # Load Test Data
            #
            with open(os.path.join(temp_dir, data_type, 'pickled_data_test.bin'), 'rb') as pk:
                test_data = pickle.load(pk)
            test_size = len(test_data)
            dataset = tf.data.Dataset.from_generator(
                make_generator(test_data),
                output_types=({'image': tf.float64, 'tableur': tf.float64}, tf.float64),
                output_shapes=({'image': (*input_shape, 12), 'tableur': 5}, 4)
            ).batch(test_size)

            #
            # Load NN Model and Predict
            #
            model: keras.Model
            model = load_model(os.path.join(model_dir, data_type, 'model'))
            y_pred = model.predict(dataset)
            y_real_findings = [x['label'] for x in test_data]
            y_real_mvi = [x['mvi'] for x in test_data]

            #
            # Estimate NN Model by Test Data
            #
            y_pred.tolist()
            y_pred = [[1 if x == max(pred) else 0 for x in pred] for pred in y_pred]
            y_real_findings = [[0 if i <= 1e-4 else 1 for i in x.tolist()] for x in y_real_findings]

            # 작성

            if os.path.exists(os.path.join(plot_dir, 'estimation_report.txt')):
                mode = 'a'
            else:
                mode = 'w'
            with open(os.path.join(plot_dir, 'estimation_report.txt'), mode) as f:
                f.write("{0} Phase Test Accuracy: ".format(data_type)
                        + str(accuracy_score(y_real_findings, y_pred)) + '\n')

            #
            # Convert Predict result to finding index
            #
            finding_index = [(0, 0), (0, 1), (1, 0), (1, 1)]
            y_pred = [finding_index[x.index(max(x))] for x in y_pred]

            temp_pred.append(y_pred)
            temp_real.append(y_real_mvi)

        #
        # Predict MVI from ML Model with Test Data
        #
        pred_findings = [x + y for x, y in zip(*temp_pred)]  # Concat Finding Predictions

        with open(os.path.join(temp_dir, 'HBP', 'pickled_data_test.bin'), 'rb') as pk:
            test_data = pickle.load(pk)

        temp_df = []
        for record_fin, record_tab in zip(pred_findings, test_data):
            if not list(record_fin) == [0, 0, 0, 0]:
                print(record_fin)
            record = record_tab['tableur'].tolist() + list(record_fin)
            temp_df.append(record)

        temp_df = pd.DataFrame(temp_df,
                               columns=['gender', 'age', 'cirrhosis', 'Tumor_size1', 'tumor(20min)',
                                        'peritumoral enhancement(arterial)', 'rim-like enhancement(arterial)',
                                        'peritumoral hypointensity(20min)', 'Irregular margin(20min)'])

        target = [1 if x > 0.5 else 0 for x in temp_real[0]]

        # Load ML Model and Predict MVI
        with open(os.path.join(model_dir, 'model.xgb'), 'rb') as pk:
            classifier: XGBClassifier
            classifier = pickle.load(pk)
        pred_y = classifier.predict(temp_df)

        #
        # Plot Confusion Matrix, Report Accuracy, Recall, F1 Score
        #
        cfm = confusion_matrix(target, pred_y)
        acc = accuracy_score(target, pred_y)
        rc = recall_score(target, pred_y)
        f1 = f1_score(target, pred_y)

        with open(os.path.join(plot_dir, 'estimation_report.txt'), 'a') as f:
            f.write('CFM: ' + str(cfm) + '\n')
            f.write('ACC' + str(acc) + '\n')
            f.write('RC' + str(rc) + '\n')
            f.write('F1' + str(f1) + '\n')

    else:
        print('Plot Data Checked')


if __name__ == '__main__':
    main(raw_path={
        'AP': sys.argv[1],
        'HBP': sys.argv[2]
    },
        data_path=sys.argv[3],
        temp_dir=sys.argv[4],
        model_dir=sys.argv[5],
        plot_dir=sys.argv[6],
        base=sys.argv[7],
        data_shape=sys.argv[8]
    )
