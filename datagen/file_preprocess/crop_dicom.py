"""
Cut dicom_png by mask_nifti_png
made by sangrae kim

input format(DICOM): ./dicom/*PATIENT_ID*/HBP/*.png
input format(NIFTI): ./mask_png/*PATIENT_ID*/*.png
output format: ./crop/*PATIENT_ID*/*.png (only layer that has mask)

When tumor size < 96*96 --> Cut by fixed size (96*96)
When tumor exceed 96*96 --> Cut by relative to its size

parameters:
    __MARGIN__: fixed margin of the mask in pixels(<96*96)
    __RESIZE__: resize the exceed image(>96*96)
    __MAX_NUM__: maximum number of images to be cropped (for multi channel train data)
"""


from PIL import Image, ImageOps
import os
import nibabel as nib
import numpy as np
import pydicom as pdc
from tqdm.auto import tqdm

__MARGIN__ = 96 / 2
__MAX_NUM__ = 0
__RESIZE__ = (96, 96)

def find_edge(mask):
    imgArray = np.array(mask)
    y_index = []
    x_index = []
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    for i in range(mask.height):
        for j in range(mask.width):
            if imgArray[i, j] >= 128:
                y_index.append(i)
                x_index.append(j)

    # if y_index & x_index is empty, means no mask, zero padding needed
    if len(y_index) == 0 or len(x_index) == 0:
        return (0, 0, 0, 0)
    

    x_min = min(x_index)
    x_max = max(x_index)
    y_min = min(y_index)
    y_max = max(y_index)
    return (x_min, x_max, y_min, y_max)


def getMask(filepath):
    img = nib.load(filepath)
    image_array = np.asanyarray(img.dataobj)
#    if(image_array == np.zeros((image_array.shape[0], image_array.shape[1]))):
#        return None
    total_slices = image_array.shape[2]
    slice_counter = 0

    for current_slice in range(0, total_slices):
        # alternate slices
        if (slice_counter % 1) == 0:
            data = image_array[:, :, current_slice]
            img = Image.fromarray(data)
            img = ImageOps.contain(img, (256, 256), Image.NEAREST)
            # alternate slices and save as png
            if (slice_counter % 1) == 0:
                slice_counter += 1
                yield img

def getPNG(filepath):
    img = Image.open(filepath)
    return img


def getDicom(filepath):
    dcm = pdc.dcmread(filepath)
    image_array = np.asanyarray(dcm.pixel_array)
    total_slices = image_array.shape[0]
    slice_counter = 0

    for current_slice in range(0, total_slices):
        # alternate slices
        if (slice_counter % 1) == 0:
            data = image_array[current_slice]
            img = Image.fromarray(data)
            # img = ImageOps.contain(img, (256, 256), Image.NEAREST)

            # alternate slices and save as png
            if (slice_counter % 1) == 0:
                slice_counter += 1
                yield img


path = os.getcwd()

for file in sorted(os.listdir(path+"/mask")):
    # if no dicom folder exists, continue
    if not os.path.exists(path+"/dicom/"+file[:-4]):
        continue
    if not os.path.exists(path+"/mask_png/"+file[:-4]):
        continue

    current_slice = 0


    for (png, mask) in tqdm(list(zip(sorted(os.listdir(path+"/dicom/"+file[:-4]+"/HBP/")), sorted(os.listdir(path+"/mask_png/"+file[:-4]+"/"))))):
        png_img = getPNG(path+"/dicom/"+file[:-4]+"/HBP/"+png)
        mask_img = getPNG(path+"/mask_png/"+file[:-4]+"/"+mask)

        x_min, x_max, y_min, y_max = find_edge(mask_img)
        
        if x_min == 0 and x_max == 0 and y_min == 0 and y_max == 0:
            continue
        

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        # if tumor size exceeds 96*96
        if x_max-x_min > 96 or y_max-y_min > 96:
            # cut by relative
            max_length = max(x_max-x_min, y_max-y_min) / 2
            mrg = 8
            png_crop = png_img.crop((x_mid-max_length-mrg, y_mid-max_length-mrg, x_mid+max_length+mrg, y_mid+max_length+mrg))
            png_crop.thumbnail(__RESIZE__, Image.LANCZOS)

        else:
            # cut by absolute
            png_crop = png_img.crop((x_mid-__MARGIN__, y_mid-__MARGIN__, x_mid+__MARGIN__, y_mid+__MARGIN__))
        
        if not os.path.exists(path+"/over/"+file[:-4]):
            os.mkdir(path+"/over/"+file[:-4])
        if not os.path.exists(path+"/crop/"+file[:-4]):
            os.mkdir(path+"/crop/"+file[:-4])

        
        png_crop.save(path+"/crop/"+file[:-4]+"/"+file[:-4]+"_c"+"{:0>3}".format(str(current_slice+1))+".png")
        current_slice += 1
    
    if(current_slice > __MAX_NUM__):
        __MAX_NUM__ = current_slice
    

# make maxnum.txt that print __MAX_NUM__
with open(path+"/maxnum.txt", "w") as f:
    f.write(str(__MAX_NUM__))
    f.close()


    



