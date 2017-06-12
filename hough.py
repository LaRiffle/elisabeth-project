#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:20:58 2017

@author: ryffel
"""
# pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from utils import *

#Should have shape > 600x600
path = 'sillons_cadres/sillon14.png'

start_time = time.time()

# I. Define rough orientation
part('I. Define rough orientation')
img = load_image(path)

img = crop_image(img, how=300)
#gray = cv2.equalizeHist(gray)
show_image(img, 'Natural image')

gray = gray_image(img, with_clahe=True)
show_image(gray, 'Gray image - clahe')

threshold = get_canny_configuration(gray)
theta, nb_lines = canny_direction(img, gray, threshold['down'], threshold['up'])


# II. Precise orientation
part(' II. Precise orientation')

img = load_image(path)

img = rotate_image(img, theta)
show_image(img, 'Rotated image '+str(theta)+'°')

img = crop_image(img, how=300)
show_image(img, 'Croped image')

gray = gray_image(img, with_clahe=True)
show_image(gray, 'Gray image - clahe')

threshold = get_canny_configuration(gray)
theta_cor, nb_lines = canny_direction(img, gray, threshold['down'], threshold['up'])

# III. Count sillons
part(' III. Count sillons')

img = load_image(path)

theta = theta + theta_cor
img = rotate_image(img, theta)
show_image(img, 'Rotated image '+str(theta)+'°')

img = crop_image(img, how=300)
show_image(img, 'Croped image')

# preprare gray for canny : good
print(img.shape)
h, w, _ = img.shape
gray = []
for i in range(h):
    gray.append([np.mean(img[:, i]) for i in range(w)])
    
gray = np.uint8(np.array(gray))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

show_image(gray, 'Gray image - clahe')

threshold = get_canny_configuration(gray)
#threshold = {'down':30, 'up': 50}
_, nb_lines = canny_direction(img, gray, threshold['down'], threshold['up'])

print('There are', nb_lines, 'lines ->', int(nb_lines/2), 'sillons')

print(time.time() - start_time, 'seconds')
