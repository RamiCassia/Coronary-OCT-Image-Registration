import os
import numpy as np
import nibabel as nib
from PIL import Image

#Function takes as input the path to nii.gz file, and the maximum distance of the catheter inside the patient (in mm)

def slice_ni(path, output_path, file_name):
    
    # Load a particular nii.gz file
    file_path = os.path.join(path, file_name)
    stacked_img = nib.load(file_path) 
    
    # Convert Image object to numpy array 
    stacked_array = np.array(stacked_img.dataobj) 

    # Specify dimensions
    width = np.size(stacked_array, axis = 0)
    height = np.size(stacked_array, axis = 1)
    depth = np.size(stacked_array, axis =2)
    channels = 3

    for i in range(depth):
    
        slice_array = stacked_array[:,:,i]

        # Create an empty image
        img = np.zeros((height, width, channels), dtype=np.uint8)

        # Set the RGB values (array of 3-tuples must be converted to an array of depth 3, one for each channel)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
            
                img[y][x][0] = slice_array[x,y][0] #R
                img[y][x][1] = slice_array[x,y][1] #G
                img[y][x][2] = slice_array[x,y][2] #B
        
        # Correct image orientation
        img = np.flipud(img)
        img = np.fliplr(img)
        
        # Naming format of .png images are *_(slice number)_(position in mm).png
        im = Image.fromarray(img)
        im.save(output_path + "/" + file_name[:-7] + '_' + str(i+1) + '.png')

if __name__ == "__main__":
    PATH = "/store/DAMTP/cr661/OCTImaging/data/5_Annotated_Files/TemporaryDir"
    OUTPUT_PATH = "/store/DAMTP/cr661/OCTImaging/data/5_Annotated_Files/SplittedImages"

    for file in os.listdir(PATH):
        print(file)
        if not os.path.isdir(file):
            slice_ni(path=PATH, output_path=OUTPUT_PATH, file_name=file)
        else:
            print(f"{file} is directory.")