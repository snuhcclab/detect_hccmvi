# SNU HCC LAB Junseo Kang
# read images and patient data file, preprocess and return as data.Dataset, pickle them.
# python preprocess.py mode data_dir temp_dir patient_dir


import os.path
import shutil
import sys
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle
from itertools import product


CUT_TO_USE = 12


def preprocess(mode, data_dir, temp_dir, patient_dir):
    """
    :param mode:
    :param data_dir:
    :param temp_dir: temp_*
    :param patient_dir:
    :return:
    """
    #
    # Check and Make Directory
    #
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    #
    # Define Generator for each Mode
    #
    def hbp3d(patient: str, label: int):
        """
        :param patient: Patient File Dir
        :param label: 0 or 1, MVI
        :return: tf.train.Example
        """
        temp = []
        for cut in os.listdir(patient):
            img = Image.open(os.path.join(patient, cut)).convert('L')
            img = np.array(img).reshape((96, 96, 1)) / 255 + 0.0001  # Preprocess
            temp.append(img)
        if len(temp) > CUT_TO_USE:
            temp = temp[:CUT_TO_USE]
        elif len(temp) < CUT_TO_USE:
            temp = temp + [np.zeros(shape=(96, 96, 1)) + 0.0001 for _ in range(CUT_TO_USE - len(temp))]
        img_3d = np.concatenate(temp, axis=2)

        label = np.array([label])

        return tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img_3d.tobytes()])
            ),
            'label': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[label.tobytes()])
            )
        }))

    def hbp3d_multimodal(patient: str, label: int, patient_data: pd.DataFrame):
        """
        :param patient_data:
        :param patient: Patient File Dir
        :param label: 0 or 1, MVI
        :return: tf.train.Example
        """
        temp = []
        for cut in os.listdir(patient):
            img = Image.open(os.path.join(patient, cut)).convert('L')
            img = np.array(img).reshape((96, 96, 1)) / 255 + 0.0001  # Preprocess
            temp.append(img)
        if len(temp) > CUT_TO_USE:
            temp = temp[:CUT_TO_USE]
        elif len(temp) < CUT_TO_USE:
            temp = temp + [np.zeros(shape=(96, 96, 1)) + 0.0001 for _ in range(CUT_TO_USE - len(temp))]
        img_3d = np.concatenate(temp, axis=2)

        label = np.array([label])

        patient_id = patient[-7:]
        patient_id = patient_data.loc[patient_data['patID'].map(int) == int(patient_id)]
        tableur = patient_id[['gender', 'age', 'cirrhosis', 'Tumor_size1']].values.tolist()[0]
        tableur[0] = 0.00001 if tableur[0] == 'M' else 0.99999
        tableur[1] = tableur[1] / 110
        tableur[2] = tableur[2] + 0.00001 if tableur[2] == 0 else 0.99999
        tableur[3] = tableur[3] / 5
        tableur = np.array(tableur, float)

        return tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img_3d.tobytes()])
            ),
            'tableur': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tableur.tobytes()])
            ),
            'label': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[label.tobytes()])
            )
        }))

    def hbp2d(cut, label):
        img = Image.open(cut)
        img = np.array(img) / 255

        if label == 1:
            label = 0.9999
        else:
            label = 0.0001

        return tf.train.Example(tf.train.Feature(feature={
            'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img])
            ),
            'label': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[label])
            )
        }))

    #
    # Convert to TFRecord
    #
    if mode == 'hbp2d':
        labels = [0, 1]
        with tf.io.TFRecordWriter(os.path.join(temp_dir, temp_dir[10:] + '.tfrecord')) as writer:  # TOdo: Generalize..
            for label in labels:
                patient_list = os.listdir(os.path.join(data_dir, str(label)))
                for patient in patient_list:
                    for image in os.listdir(os.path.join(data_dir, str(label), patient)):
                        ex = hbp2d(os.path.join(data_dir, str(label), patient, image),
                                   label)
                        writer.write(ex.SerializeToString())

    if mode == 'hbp2d_multiclass':
        labels = [0, 1]
        with tf.io.TFRecordWriter(os.path.join(temp_dir, temp_dir[10:] + '.tfrecord')) as writer:
            for label in labels:
                patient_list = os.listdir(os.path.join(data_dir, str(label)))
                for patient in patient_list:
                    for image in os.listdir(os.path.join(data_dir, str(label), patient)):
                        ex = hbp2d(os.path.join(data_dir, str(label), patient, image),
                                   label)
                        writer.write(ex.SerializeToString())

    elif mode == 'hbp3d':
        labels = [0, 1]
        with tf.io.TFRecordWriter(os.path.join(temp_dir, temp_dir[10:] + '.tfrecord')) as writer:
            for label in labels:
                patient_list = os.listdir(os.path.join(data_dir, str(label)))
                for patient in patient_list:
                    ex = hbp3d(os.path.join(data_dir, str(label), patient),
                               label)
                    writer.write(ex.SerializeToString())

    elif mode == 'hbp3d_multimodal':
        labels = [0, 1]
        patient_df = pd.read_excel(patient_dir, sheet_name='data')
        with tf.io.TFRecordWriter(os.path.join(temp_dir, temp_dir[10:] + '.tfrecord')) as writer:
            for label in labels:
                patient_list = os.listdir(os.path.join(data_dir, str(label)))
                for patient in patient_list:
                    ex = hbp3d_multimodal(os.path.join(data_dir, str(label), patient),
                                          label,
                                          patient_df)
                    writer.write(ex.SerializeToString())

    return True


def preprocess_pk(mode, data_dir, temp_dir, patient_dir):
    #
    # Check and make Directory
    #
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    #
    # Define Generator of each mode
    #
    def hbp3d(patient: str, label: int):
        """
        :param patient: Patient File Dir
        :param label: 0 or 1, MVI
        :return: tf.train.Example
        """
        temp = []
        for cut in os.listdir(patient):
            img = Image.open(os.path.join(patient, cut)).convert('L')
            img = np.array(img).reshape((96, 96, 1)) / 255 + 0.0001  # Preprocess
            temp.append(img)
        if len(temp) > CUT_TO_USE:
            temp = temp[:CUT_TO_USE]
        elif len(temp) < CUT_TO_USE:
            temp = temp + [np.zeros(shape=(96, 96, 1)) + 0.0001 for _ in range(CUT_TO_USE - len(temp))]
        img_3d = np.concatenate(temp, axis=2)

        label = np.array([label])

        return {'image': img_3d,
                'label': label}

    def hbp3d_multimodal(patient: str, label: int, patient_data: pd.DataFrame):
        """
        :param patient_data:
        :param patient: Patient File Dir
        :param label: 0 or 1, MVI
        :return: tf.train.Example
        """
        temp = []
        for cut in os.listdir(patient):
            img = Image.open(os.path.join(patient, cut)).convert('L')
            img = np.array(img).reshape((96, 96, 1)) / 255 + 0.0001  # Preprocess
            temp.append(img)
        if len(temp) > CUT_TO_USE:
            temp = temp[:CUT_TO_USE]
        elif len(temp) < CUT_TO_USE:
            temp = temp + [np.zeros(shape=(96, 96, 1)) + 0.0001 for _ in range(CUT_TO_USE - len(temp))]
        img_3d = np.concatenate(temp, axis=2)

        label = np.array([label])

        patient_id = patient[-7:]
        patient_id = patient_data.loc[patient_data['patID'].map(int) == int(patient_id)]
        tableur = patient_id[['gender', 'age', 'cirrhosis', 'Tumor_size1']].values.tolist()[0]
        tableur[0] = 0.00001 if tableur[0] == 'M' else 0.99999
        tableur[1] = tableur[1] / 110 + 0.00001
        tableur[2] = tableur[2] + 0.00001 if tableur[2] == 0 else 0.99999
        tableur[3] = tableur[3] / 5
        tableur = np.array(tableur, float)

        return {'image': img_3d,
                'tableur': tableur,
                'label': label}

    def concat3d_wjl(patient: str, label: int, patient_data: pd.DataFrame):
        """
        :param patient:
        :param label:
        :param patient_data:
        :return:
        """
        temp = []
        patient_id = patient[-7:]
        patient_id = patient_data.loc[patient_data['patID'].map(int) == int(patient_id)]
        if not len(patient_id['tumor(20min)']) == 0:
            tumor_cut = [int(x) for x in patient_id['tumor(20min)'].str.split(',').tolist()[0]]
        else:
            return False

        for i, cut in enumerate(os.listdir(patient)):
            if tumor_cut[0] <= i <= tumor_cut[1]:
                img = Image.open(os.path.join(patient, cut)).convert('L')
                img = np.array(img).reshape((96, 192, 1)) / 255 + 0.0001  # Preprocess
                temp.append(img)

        if len(temp) > CUT_TO_USE:
            temp = temp[:CUT_TO_USE]
        elif len(temp) < CUT_TO_USE:
            temp = temp + [np.zeros(shape=(96, 192, 1)) + 0.0001 for _ in range(CUT_TO_USE - len(temp))]
        img_3d = np.concatenate(temp, axis=2)

        tableur = patient_id[['gender', 'age', 'cirrhosis', 'Tumor_size1']].values.tolist()[0]
        label = patient_id[['peritumoral enhancement(arterial)',
                            'rim-like enhancement(arterial)',
                            'peritumoral hypointensity(20min)',
                            'Irregular margin(20min)']].values.tolist()[0]
        machine = patient_id[['Machine']].values.tolist()[0][0]

        tableur[0] = 0.00001 if tableur[0] == 'M' else 0.99999
        tableur[1] = tableur[1] / 110 + 0.00001
        tableur[2] = tableur[2] + 0.00001 if tableur[2] == 0 else 0.99999
        if tableur[3] <= 2 or not machine.startswith('A'):
            return False
        tableur[3] = tableur[3] / 5

        tumor_z_size = (tumor_cut[1] - tumor_cut[0])/len(os.listdir(patient))
        tableur.append(tumor_z_size)

        label[0] = 0 if label[0] == 0 or label[0] == '0m' or label[0] == 'x' or label[0] == '?' else 1
        label[1] = 0 if label[1] == 0 or label[1] == '0m' or label[0] == 'x' or label[0] == '?' else 1
        label[2] = 0 if label[2] == 0 or label[2] == '0m' or label[0] == 'x' or label[0] == '?'else 1
        label[3] = 0 if label[3] == 0 or label[3] == '0m' or label[0] == 'x' or label[0] == '?' else 1
        label = tuple(label[2:])

        def make_combination(ser, n=4):
            items = [(0, 1) for _ in range(n)]
            able = list(product(*items))
            able = sorted(able,
                          key=lambda x: x.count(1))
            idx = able.index(ser)
            return [1e-5 if x != idx else 1-1e-5 for x in range(2**n)]

        label = make_combination(label, n=2)

        tableur = np.array(tableur, float)
        label = np.array(label, float)

        return {'image': img_3d,
                'tableur': tableur,
                'label': label}

    def hbp2d(cut, label):
        img = Image.open(cut)
        img = np.array(img) / 255

        if label == 1:
            label = 0.9999
        else:
            label = 0.0001

        return {
            'image': img,
            'label': label
        }

    #
    # Pickle
    #
    if mode == 'hbp3d':
        temp = []
        labels = [0, 1]
        for label in labels:
            patient_list = os.listdir(os.path.join(data_dir, str(label)))
            for patient in patient_list:
                ex = hbp3d(os.path.join(data_dir, str(label), patient),
                           label)
                temp.append(ex)
        with open(os.path.join(temp_dir, temp_dir[10:] + '.bin'), 'wb') as pk:
            pickle.dump(temp, pk)

    elif mode == 'hbp3d_multimodal':
        temp = []
        labels = [0, 1]
        patient_df = pd.read_excel(patient_dir, sheet_name='data')
        for label in labels:
            patient_list = os.listdir(os.path.join(data_dir, str(label)))
            for patient in patient_list:
                ex = hbp3d_multimodal(os.path.join(data_dir, str(label), patient),
                                      label,
                                      patient_data=patient_df)
                temp.append(ex)
        with open(os.path.join(temp_dir, temp_dir[10:] + '.bin'), 'wb') as pk:
            pickle.dump(temp, pk)

    elif mode == 'concat3d_wjl':
        temp = []
        labels = [0, 1]
        patient_df = pd.read_excel(patient_dir, sheet_name='data').dropna()
        for label in labels:
            patient_list = os.listdir(os.path.join(data_dir, str(label)))
            for patient in patient_list:
                if not patient.endswith('DS_Store'):
                    ex = concat3d_wjl(os.path.join(data_dir, str(label), patient),
                                      label,
                                      patient_data=patient_df)
                    if not ex:
                        continue
                    temp.append(ex)
        with open(os.path.join(temp_dir, temp_dir[10:] + '.bin'), 'wb') as pk:
            pickle.dump(temp, pk)

    return True


if __name__ == '__main__':
    if sys.argv[5] == 'p':
        preprocess_pk(mode=sys.argv[1],
                      data_dir=sys.argv[2],
                      temp_dir=sys.argv[3],
                      patient_dir=sys.argv[4])
    else:
        preprocess(mode=sys.argv[1],
                   data_dir=sys.argv[2],
                   temp_dir=sys.argv[3],
                   patient_dir=sys.argv[4])
