# SNU HCC LAB Junseo Kang, Sangrae Kim
# Convert Dicom-NFITI File to PNG, crop tumor image with Margin, Label by Patient Data File
# python convert_crop.py dicom_dir nifti_dir patient_dir result_dir margin

import pydicom
import os
import shutil
import sys
import numpy as np
from PIL import Image
import cv2


def main(dicom_dir, nifti_dir, result_dir, margin=0.1):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    patients = os.listdir(dicom_dir)
    for patient in patients:
        os.mkdir(os.path.join(result_dir, patient))
        for type in ('AP', 'HBP'):
            os.mkdir(os.path.join(result_dir, patient, type))
            files = os.listdir(os.path.join(dicom_dir, patient, type))
            for file in files:
                filename = file[:-3]
                dcm = pydicom.dcmread(os.path.join(dicom_dir, patient, type, file))
                img = dcm.pixel_array
                img = (img-np.min(img))/(np.max(img)-np.min(img)) * 255
                img = Image.fromarray(img).convert('RGB')
                img.save(os.path.join(result_dir, patient, type, filename+'.png'), 'png')


if __name__ == '__main__':
    main(dicom_dir=sys.argv[1],
         nifti_dir=sys.argv[2],
         result_dir=sys.argv[3])
