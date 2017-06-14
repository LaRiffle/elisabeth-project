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
import numpy.polynomial.polynomial as poly
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



# Last things to do
h, w, _ = img.shape

total_bright = [np.mean(img[:, i]) for i in range(w)]
x = list(range(len(total_bright)))
coefs = poly.polyfit(x, total_bright, 5)
y_base = poly.polyval(x, coefs)
plt.figure(1)
plt.plot(x, total_bright, 'r--')
plt.plot(x, y_base)
plt.show()

y = total_bright - y_base
y = [i+128 for i in y]
plt.figure(2)
plt.plot(range(len(total_bright)), y, 'r--')
plt.show()

plt.plot(range(w), [np.mean(img[:, i]) for i in range(w)])
plt.title('Intensity profile')
plt.show()

plt.plot(range(w), [softmax(i) for i in y])
plt.title('New intensity profile')
plt.show()

detection = [softmax(i) for i in y]

fringes = []
for i in range(len(detection)):
    # saving start of anomaly with default end
    if detection[i] > 128 and (i == 0 or detection[i-1] <= 128):
        fringes.append([i, len(detection)-1])
    # saving end of anomaly in last elmt
    if detection[i] <= 128 and detection[i-1] > 128:
        if(len(fringes)-1 >= 0):
            fringes[len(fringes)-1][1] = i
        
fringes = [round((e[0]+e[1])/2) for e in fringes]      
print(fringes)

print('There are', len(fringes), 'sillons')

 nb_lines = canny_direction(img, gray, threshold['down'], threshold['up'])

print(time.time() - start_time, 'seconds')
