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
path = 'sillons_cadres/sillon26.png'
#path = 'sillons_2/sillon19.jpg'
def get_furrows_gap(path, verb=''):
    start_time = time.time()
    
    # I. Define orientation
    if 'v' in verb:
        part('I. Define orientation')
    img = load_image(path)
    
    theta = find_rotation(img, verb)
    
    
    # II. Count sillons
    if 'v' in verb:
        part('II. Count sillons')
    img = load_image(path)
    
    img = rotate_image(img, theta)
    # show_image(img, 'Rotated image '+str(theta)+'°')
    
    img = crop_image(img, how=300)
    if 'v' in verb:
        show_image(img, 'Croped image')
    
    
    
    # Last things to do
    h, w, _ = img.shape
    
    total_bright = [np.mean(img[:, i]) for i in range(w)]
    
    # Polyfit and exacerbate have the same role, but polyfit 
    # excels in detecting gap between songs et exarcerbate in detecting
    # small furrows
    detection = polyfit(total_bright, verb)
    fringes = compute_fringes_from_profile(detection)
    
    ## detection of gap betwen songs
    if has_change_of_song(fringes) is not False:
        end_song, start_song = has_change_of_song(fringes)
        song_gap = start_song - end_song
        if 'v' in verb:
            print('gap detected', end_song, start_song, song_gap)
        detection1 = exacerbate(total_bright[:end_song], verb)
        fringes1 = compute_fringes_from_profile(detection1)
        fringes1 = [round((e[0]+e[1])/2) for e in fringes1] 
        detection2 = exacerbate(total_bright[start_song:], verb)
        fringes2 = compute_fringes_from_profile(detection2)
        fringes2 = [round((e[0]+e[1])/2) for e in fringes2]
        if 'v' in verb:
            print('There are', len(fringes1), 'sillons in the last song')
            print(fringes1)
            print('There is a change of song')
            print('There are', len(fringes2), 'sillons in the new song')
            print(fringes2)
        gaps = fringes_to_gaps(fringes1) + [song_gap] + fringes_to_gaps(fringes2)
    
    else:
        detection = exacerbate(total_bright, verb)
        fringes = compute_fringes_from_profile(detection)
        fringes = [round((e[0]+e[1])/2) for e in fringes]  
        # fringes = remove_forks(fringes)    
        if 'v' in verb:
            print(fringes)
            print('There are', len(fringes), 'sillons')
        gaps = fringes_to_gaps(fringes)    
        
    if 'v' in verb:
        print(time.time() - start_time, 'seconds')
    print(gaps)
    return gaps

if __name__ == '__main__':
    get_furrows_gap(path, verb='v')