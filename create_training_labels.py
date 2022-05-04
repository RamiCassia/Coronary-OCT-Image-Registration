import os
import nibabel as nib
import numpy as np
import csv

# Set path where to find the annotated NIFTI files and where to put the csv output
img_path = "/store/DAMTP/cr661/OCTImaging/data/5_Annotated_Files/AnnotatedNIFTIFiles/"
output_path = "/home/mm2559/Documents/OCTImaging/"
output_filename = "branch_labels.csv"

# Find files with sidebranch annotation
filenames = next(os.walk(img_path), (None, None, []))[2]
sidebranch_imgs = [fn for fn in filenames if ("branch" in fn and ".nii" in fn)]


# create the lists which are later converted into csv
#   names:  contains the file names
#   slice:  contains the running index of the slice
#   labels: 0 if no information in that slice of that file (i.e. all 0s), 1 otherwise
final_list_names = []
final_list_slice = []
final_list_labels = []

for i in range(len(sidebranch_imgs)): # cycle through nifti files

    img = nib.load(os.path.join(img_path, sidebranch_imgs[i]))
    number_slices = np.shape(img)[2]

    final_list_names = final_list_names + [sidebranch_imgs[i]] * number_slices
    final_list_slice = final_list_slice + list(range(number_slices))

    img_data = img.get_fdata()
    temp_list = [None] * number_slices

    for j in range(number_slices): # cycle through layers/slices/images in each nifti file
        temp_list[j] = int(np.count_nonzero(img_data[:,:,j]) > 0)

    final_list_labels = final_list_labels + temp_list
    

# prepare to write csv
output_data = list(zip(final_list_names, final_list_slice, final_list_labels))

# write csv
header = ["filename", "slice", "branch"]

with open(os.path.join(output_path, output_filename), "w", encoding="UTF8", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(output_data)

