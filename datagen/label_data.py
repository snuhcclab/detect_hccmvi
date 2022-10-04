# SNU HCC LAB Junseo Kang
# Label Data; Patient_ID/imgs.. to label/pid/imgs..
# python label_data.py raw_dir result_dir patient_file
import os
import sys
import pandas as pd
import shutil


def main(raw_dir, result_dir, patient_file, mode=None):
    #
    # Prepare Directory
    #
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    #
    # Read Patient File
    #
    patient = pd.read_excel(patient_file, sheet_name='data')
    ava_patients = os.listdir(raw_dir)

    #
    # For Each Case
    #
    case = [0, 1]

    for each_case in case:
        os.mkdir(os.path.join(result_dir, str(each_case)))
        #
        # Check Applicable Patients
        #

        def pad(dig: int, length):
            d = str(dig)
            if len(str(d)) < length:
                return '0' * (length-len(d)) + d
            else:
                return d

        mvi_app = patient[patient['MVI'] == each_case]['patID'].map(lambda x: pad(x, 7))

        #
        # Copy Them to Result Directory
        #
        for mvi_t in mvi_app:
            if mvi_t in ava_patients:
                shutil.copytree(os.path.join(raw_dir, os.path.join(mvi_t, mode) if mode else mvi_t),
                                os.path.join(result_dir, str(each_case), mvi_t))

    return 0


if __name__ == '__main__':
    main(raw_dir=sys.argv[1],
         result_dir=sys.argv[2],
         patient_file=sys.argv[3],
         mode=None)
