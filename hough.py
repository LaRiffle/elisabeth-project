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
path = 'sillons_cadres/sillon23.png'
path = 'sillons_2/sillon20.jpg'

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
#show_image(img, 'Rotated image '+str(theta)+'°')

img = crop_image(img, how=300)
#show_image(img, 'Croped image')



# Last things to do
h, w, _ = img.shape

total_bright = [np.mean(img[:, i]) for i in range(w)]

# Polyfit and exacerbate have the same role, but polyfit 
# excels in detecting gap between songs et exarcerbate in detecting
# small furrows
detection = polyfit(total_bright)
fringes = compute_fringes_from_profile(detection)

## detection of gap betwen songs
if has_change_of_song(fringes) is not False:
    end_song, start_song = has_change_of_song(fringes)
    print('gap detected', end_song, start_song, start_song - end_song)
    print('END OF SONG')
    detection1 = exacerbate(total_bright[:end_song])
    fringes1 = compute_fringes_from_profile(detection1)
    fringes1 = [round((e[0]+e[1])/2) for e in fringes1] 
    print('START OF SONG')
    detection2 = exacerbate(total_bright[start_song:])
    fringes2 = compute_fringes_from_profile(detection2)
    fringes2 = [round((e[0]+e[1])/2) for e in fringes2]
    print('There are', len(fringes1), 'sillons in the last song')
    print('There is a change of song')
    print('There are', len(fringes2), 'sillons in the new song')
    
else:
    detection = exacerbate(total_bright)
    fringes = compute_fringes_from_profile(detection)
    fringes = [round((e[0]+e[1])/2) for e in fringes]  
    #fringes = remove_forks(fringes)    
    print(fringes)
    
    
    print('There are', len(fringes), 'sillons')

# nb_lines = canny_direction(img, gray, threshold['down'], threshold['up'])

print(time.time() - start_time, 'seconds')
