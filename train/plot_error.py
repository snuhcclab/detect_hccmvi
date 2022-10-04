# SNU HCC LAB Junseo Kang
#
#
import sys

from keras.models import load_model
import os
import pandas as pd
import tensorflow as tf
import pickle

DATA_HOME = 'data'

# Read Model, Read Data, Plot Outputs


def main(model_dir, data_dir, patient_dir, result_dir):
    #
    # Read Data Dir
    #

    def decode_to_generator(dataset_name, scheme='original'):
        with open(data_dir, 'rb') as pk:
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

    dataset = tf.data.Dataset.from_generator(
        decode_to_generator('hpb3d_padding_multimodal', scheme='multimodal'),
        output_types=({'image': tf.float64, 'tableur': tf.float64}, tf.float64),
        output_shapes=({'image': (96, 96, 16), 'tableur': (4)}, 1)
    ).batch(8)

    patients = os.listdir('data/hbp3d/0') + os.listdir('data/hbp3d/1')


    #
    # Read Model
    #
    model = load_model(model_dir)

    #
    # Estimate
    #
    print(model.predict(dataset).tolist())


if __name__ == '__main__':
    main(model_dir=sys.argv[1],
         data_dir=sys.argv[2],
         patient_dir=sys.argv[3],
         result_dir=sys.argv[4])
