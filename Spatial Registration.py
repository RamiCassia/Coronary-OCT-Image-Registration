import numpy as np
import cv2
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import os
import sys
from skimage import transform, io, exposure
from pystackreg import StackReg #pystackreg is used to perform spatial image registration
import pystackreg
import scipy as sc

#Utility for removing timestap at the top right of OCT image, and logo at bottom right of OCT image

def remove_artifacts(image):

  if np.ndim(image) == 3:
    #Remove text on top right
    image[:75,800:-1,0] = 0
    image[:75,800:-1,1] = 0
    image[:75,800:-1,2] = 0

    #Remove logo on bottom right
    image[900:-1,930:-1,0] = 0
    image[900:-1,930:-1,1] = 0
    image[900:-1,930:-1,2] = 0
  else:
    image[:75,800:-1] = 0
    image[900:-1,930:-1] = 0

  return np.array(image, dtype = np.uint8)

#Utility for overlaying images

def composite_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = [img / img.max() for img in imgs]

    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))

    imgs = np.dstack(imgs)

    return imgs

#Spatial registration function. Inputs are the paths to the two images and the required transformation

def spatial_registration(path_BL, path_FUP, transformation):

  ref = io.imread(path_BL) #Read baseline image
  mov = io.imread(path_FUP) #Read follow-up image

  ref = remove_artifacts(ref) #Remove artifacts of baseline image (i.e. remove the time stamo and logo)
  mov = remove_artifacts(mov) #Remove artifacts of follow-up image (i.e. remove the time stamo and logo)

  ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) #Convert baseline image to grayscale
  mov = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY) #Convert follow-up image to grayscale


  np.random.seed(0)
  random_angle = np.random.randint(0, 360, (100)) #Generate 100 random angles for rotating follow-up image (prevents trapping in local minima)

  residues = []
  errors = []

  lowest_residue = 1000000

  for n in range(50): #For each random angle in random_angle...
    mov_rot = sc.ndimage.rotate(mov, random_angle[n], reshape = False) #...rotate follow-up image by the random angle

    f, ax = plt.subplots(1, 1, figsize=(8, 8)) #Prepare figure

    for i, (name, tf) in enumerate(transformation.items()):
        sr = StackReg(tf)
        reg = sr.register_transform(ref, mov_rot) #Perfrom the registration between the baseline and randomly rotated follow-up
        reg = reg.clip(min=0)

        after_reg =  composite_images([ref, reg, mov]) #Create overlay of baseline, transfomed follow-up, and follow-up

        residue = [np.round((np.linalg.norm(ref/255 - reg/255))**2,1)] #Calculate pixel-normalized residue
        residues = residues + residue #Append residue to the history
        lowest_residue = min(residues) #Find the lowest residue achieved so far

        error = [np.round((np.linalg.norm(ref - reg))**2/(np.linalg.norm(ref))**2,2)] #Calculate error percentage between the baseline and transformed follow-up
        errors = errors + error #Append error to the history
        lowest_error = min(errors) #Find the lowest error achieved so far

        ax.imshow(after_reg, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name + ', ' + 'Residue' + '=' + str(residue[0]))
        ax.axis('off')
        f.tight_layout() #Plot overlay image after_reg

  print('Lowest residue: ' + str(lowest_residue)) #Print lowest residue achieved
  print('Lowest error: ' + str(lowest_error)) #Print lowest error achieved

#Example use of spatial_registration function

transformation = {
   #'Translation': StackReg.TRANSLATION,
   #'Rigid Body': StackReg.RIGID_BODY,
   #'SCALED_ROTATION': StackReg.SCALED_ROTATION,
   'General Affine': StackReg.AFFINE,
   #'BILINEAR': StackReg.BILINEAR
}

path_BL = '/path/to/BL/image.png'
path_FUP ='/path/to/FUP/image.png'

spatial_registration(path_BL, path_FUP, transformation)