"""
python script to convert nifti to png
made by sangrae kim

input format(nii.gz): ./nii/*.nii.gz
output format(png): ./nii2png/*PATIENT_ID*/*.png

parameters:
    path_dir: the directory where the NIfTI files are located
    output_dir: the directory where the PNG files will be created
    invt: invert the image(0->255, 255->0)
"""
import numpy as np
import os
import nibabel as nib
from PIL import Image, ImageOps
from tqdm.auto import tqdm

def convert(inputfile, outputfile, invt=False):
    """
    Convert a NIfTI file to a PNG file.
    inputfile: the NIfTI file to convert
    outputfile: dir where the PNG file to create
    """


    img = nib.load(inputfile)
    image_array = np.asanyarray(img.dataobj)
    if len(image_array.shape) == 3:

        # make output directory if it doesn't exist
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)

        total_slices = image_array.shape[2]

        slice_counter = 0

        # iterate through slices
        for current_slice in range(0, total_slices):
            # alternate slices
            if (slice_counter % 1) == 0:

                data = image_array[:, :, current_slice]
                img = Image.fromarray(data)
                if data.shape[0] != data.shape[1]:
                    pass
                    img = img.rotate(90, expand=1).transpose(Image.FLIP_TOP_BOTTOM)
                    #img.show()
                else:
                    img = img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
                    #img.show()
                

                #alternate slices and save as png
                if (slice_counter % 1) == 0:
                        # image original(not inverted)
                    if not invt:
                        image_name = outputfile + "/" + inputfile[-11:-4] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                        img.save(image_name)
                        
                        slice_counter += 1

                        # image invert
                    else:
                        image_name = outputfile + "/" + inputfile[-11:-4] + "_t" + "{:0>3}".format(str(current_slice+1))+ ".png"
                        img = ImageOps.invert(img)
                        img.save(image_name)
                        
                        slice_counter += 1


    else:
        print('Not a 3D image. Please try again.')




path_dir = os.getcwd()
path_dir, output_dir = path_dir+'/nii/', path_dir+'/nii2png/'
leave = input("Do you want to inverted the image? (y/n): ")
invt = True if leave.lower() == 'y' else False

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Created ouput directory: " + output_dir)


for file in tqdm(sorted(os.listdir(path_dir)),desc='Converting NIFTI to PNG', leave=True):
    if file.endswith(".nii"):
        convert(path_dir+file, output_dir+file[:-4], invt)
        # print('Converted '+file)
    else:
        continue

print('Finished converting NIFTI to PNG')

