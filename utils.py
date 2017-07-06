#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import time
import math
from copy import deepcopy

MIN_GAP_SONG = 30
AVG_WIDTH_FURROW = 16

    
def polyfit(total_bright, verb=''):
    '''
        Fits profile with polynoms to detect large gaps => change in songs
    '''
    x = list(range(len(total_bright)))
    coefs = poly.polyfit(x, total_bright, 5)
    y_base = poly.polyval(x, coefs)
    if 'v' in verb:
        plt.figure(1)
        plt.plot(x, total_bright, 'r--')
        plt.plot(x, y_base)
        plt.title('Polyfit')
        plt.show()
    
    y = total_bright - y_base
    y = [i+128 for i in y]
    
    detection = [softmax(i) for i in y]

    return detection
    
def exacerbate(total_bright, verb=''):
    """
        Fits closely profile to exacerbate furrows
    """
    x = list(range(len(total_bright)))
    
    y_base = []
    width = int(AVG_WIDTH_FURROW/2)
    for i in range(len(total_bright)):
        y_base.append(np.mean(total_bright[max(0, i-width):min(len(total_bright)-1, i+width)]))
    
    if 'v' in verb:    
        plt.figure(1)
        plt.title('Local means (exacerbate)')
        plt.plot(x, total_bright, 'r--')
        plt.plot(x, y_base)
        plt.show()
    
    y = [tot - base for tot, base in zip(total_bright, y_base)]
    y = [i+128 for i in y]

    if 'v' in verb:
        plt.figure(2)
        plt.title('Intensity profile (exacerbate)')
        plt.plot(range(len(total_bright)), y)
        plt.show()
        
        plt.plot(range(len(total_bright)), [softmax(i) for i in y])
        plt.title('New intensity profile (exacerbate)')
        plt.show()
    
    detection = [softmax(i) for i in y]

    return detection
    
def compute_fringes_from_profile(detection):
    fringes = []
    for i in range(len(detection)):
        # saving start of anomaly with default end
        if detection[i] > 128 and (i == 0 or detection[i-1] <= 128):
            fringes.append([i, len(detection)-1])
        # saving end of anomaly in last elmt
        if detection[i] <= 128 and detection[i-1] > 128:
            if(len(fringes)-1 >= 0):
                fringes[len(fringes)-1][1] = i
    return fringes
    
def has_change_of_song(fringes):
    for fringe1, fringe2 in zip(fringes[:-1], fringes[1:]):
        if fringe2[0] - fringe1[1] > MIN_GAP_SONG:
            return (fringe1[1], fringe2[0])
    return False
    
def remove_forks(fringes):
    ecart = [fringes[i+1]-fringes[i] for i in range(len(fringes)-1)]
    mean = np.mean(ecart)
    forks = [ i  for i, e in enumerate(ecart) if e < 2*mean/3]
    pos = fringes[0]
    fringes = [fringes[0]]
    for i, e in enumerate(ecart):
        if i+1 in forks:
            pos += (ecart[i] + int(ecart[i+1]/2))
            fringes.append(pos)
        elif i in forks:
            pass
        elif i-1 in forks:
            pos += (ecart[i] + ecart[i-1] - int(ecart[i-1]/2))
            fringes.append(pos)
        else:
            pos += ecart[i]
            fringes.append(pos)
    return fringes
    
def load_image(path):
    return cv2.imread(path)
    

def show_image(img, title=''):
    plt.imshow(img)
    plt.title(title)
    plt.show()
    print(img.shape)
    
def rotate_image(img, theta):
    rows,cols,rgb = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
    
def crop_image(img, how='half'):
    if how is 'half':
        rows,cols,rgb = img.shape
        crop_img = img[int(0.25*rows):int(0.75*rows), int(0.25*cols):int(0.75*cols)] # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        return crop_img
    else:
        rows,cols,rgb = img.shape
        crop_img = img[int(0.5*rows - how/2):int(0.5*rows + how/2), int(0.5*cols - how/2):int(0.5*cols + how/2)] # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        return crop_img
    
def softmax(x):
    return 255/(1+math.exp(-(x-128)))
    
def part(title):
    print()
    print(title)
    print()
    
# see if the image is correctly oriented : var max    
def compute_variance(total_bright):
    return np.var(total_bright)
    
def find_rotation(img, verb=''):
    steps = [1, 0.1]
    theta_extremum = [40, 110] # 0,180 you can reduce if you have a basic estimate
    for step in steps:
        theta_range = np.arange(theta_extremum[0], theta_extremum[1], step)
        is_first = (step is steps[0])
        best_theta = get_rotation_with_higher_variance(img, theta_range, step, is_first, verb)
        theta_extremum = [best_theta - 2*step, best_theta + 2*step]
    if 'v' in verb:  
        print('Best', best_theta)
    return best_theta
    
def get_rotation_with_higher_variance(img, theta_range, step, is_first, verb):
    """
        Returns the theta for which a singularity has been detected
    """
    var_range = []
    for theta in theta_range:
        variance = compute_variance_of_rotated_img(img, theta)
        var_range.append(variance)
    # print(var_range)
    
    # Detect singularity which is not always the argmax
    # Plot for more details
    if is_first:
        y_base = []
        width = 5
        for i in range(len(var_range)):
            y_base.append(np.mean(var_range[max(0, i-width):min(len(var_range)-1, i+width)]))
        
        x = theta_range
        if 'v' in verb: 
            plt.figure(1)
            plt.title('Fitting the global shape')
            plt.plot(x, var_range)
            plt.plot(x, y_base, 'r--')
            plt.show()
        
        var_range = [var - base for var, base in zip(var_range, y_base)]
        # cut extremmities
        for i, var in enumerate(var_range):
            if i < width or i >= len(var_range) - width:
                var_range[i] = 0

    best_theta = theta_range[np.argmax(var_range)]
    return best_theta
    
def compute_variance_of_rotated_img(img, theta):
    img2 = deepcopy(img)
    img2 = rotate_image(img2, theta)
    img2 = crop_image(img2, how=300)
    h, w, _ = img2.shape
    total_bright = [np.mean(img2[:, i]) for i in range(w)]
    return compute_variance(total_bright)


def fringes_to_gaps(fringes):
    l_left = fringes[:-1]
    l_right = fringes[1:]
    gaps = []
    for i1, i2 in zip(l_left, l_right):
        gap = i2 - i1
        gaps.append(gap)
    return gaps
    
