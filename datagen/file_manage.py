# SNU HCC LAB Junseo Kang
#
# python convert_crop.py dicom_dir nifti_dir patient_dir result_dir margin
import sys
import pandas as pd
import shutil
import os
import re


def main(raw_dir, data_dir, new_dir):
    """
    :param raw_dir:
    :param data_dir:
    :param new_dir:
    :return:

    1) eTHRIVE A, eTHRIVE 20MIN
    2) t1_vibe_fs_tra_bh_dynamic, t1_vibe_fs_tra_bh_20min
    3) sWIP 4D THRIVE A-P 1, eTHRIVE 20MIN
    4) T1 VIBE CAIPI FS ARTERIAL, T1 VIBE CAIPI FS 20MIN
    """
    #
    # Read Patient Data
    #
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)
    patient_id = pd.read_excel(data_dir, sheet_name='data')["patID"].tolist()

    #
    # Check MRI, select AP-HBP, Copy to $new_dir
    #
    patient_list = os.listdir(raw_dir)
    total_count = len(patient_list)
    count = 0
    for patient in patient_list:
        ap = None
        hbp = None
        mri_folder = None
        if int(patient) in patient_id:
            # Check MRI
            folders = os.listdir(os.path.join(raw_dir, patient))
            for folder in folders:
                files = os.listdir(os.path.join(raw_dir, patient, folder))
                if "PET" in ''.join(files) or "CT_tr" in ''.join(files):
                    continue
                else:
                    mri_folder = folder

                conditions = [
                    (re.compile("eTHRIVE A"), re.compile("eTHRIVE 20MIN")),
                    (re.compile("t1_vibe_fs_tra_bh_dynamic"), re.compile("t1_vibe_fs_tra_bh_20min")),
                    (re.compile("sWIP 4D THRIVE A-P 1"), re.compile("eTHRIVE 20MIN")),
                    (re.compile("T1 VIBE CAIPI FS ARTERIAL"), re.compile("T1 VIBE CAIPI FS 20MIN"))
                ]
                for file in files:
                    for condition in conditions:
                        if condition[0].search(file) and not ap:
                            ap = file
                        elif condition[1].search(file) and not hbp:
                            hbp = file
            if not (ap and hbp and mri_folder):
                print("Error Occurred at file {0}".format(patient))
            else:
                os.mkdir(os.path.join(new_dir, patient))
                shutil.copytree(os.path.join(raw_dir, patient, mri_folder, ap), os.path.join(new_dir, patient, 'AP'))
                shutil.copytree(os.path.join(raw_dir, patient, mri_folder, hbp), os.path.join(new_dir, patient, 'HBP'))
                count += 1
    print("Total {0} Data Processed from {1} Origin Data".format(count, total_count))
    return 0


if __name__ == '__main__':
    main(raw_dir=sys.argv[1],
         data_dir=sys.argv[2],
         new_dir=sys.argv[3])
