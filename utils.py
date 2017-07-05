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


def get_canny_configuration(gray):
    best_score = 0
    threshold = {'down':0, 'up':0}
    for down in np.arange(0, 241, 10):
        for width in np.arange(20, 221, 10):
            if down + width <= 255:
                threshold_down = down
                threshold_up = min(down + width, 255)
                edges = cv2.Canny(gray,threshold_down,threshold_up,apertureSize = 3)
                lines = cv2.HoughLines(edges,1,np.pi/180,80)
                if lines is None:
                    lines = []
                lines = [list(e[0]) for e in lines]
                direction = []
                for rho,theta in lines:
                    theta = (theta + math.pi/2)%math.pi - math.pi/2
                    direction.append(theta)
                my_direction = np.mean(direction)*180/np.pi
                my_std = np.std(direction)*180/np.pi
                if len(lines) < 3:
                    score = 0
                else:
                    my_std = max(1, my_std)
                    score = min(len(lines), 50)/my_std
                #if len(lines) < 1000 and len(lines) > 0:
                #    print('conf', threshold_down, threshold_up, len(lines), 'lines', 'dir :', my_direction, '±', my_std, 'score', score)
                if score > best_score:
                    best_score = score
                    threshold['down'] = threshold_down
                    threshold['up'] = threshold_up
    print('Recommended', threshold, best_score)
    return threshold
            

def canny_direction(img, gray,threshold_down, threshold_up):
    edges = cv2.Canny(gray,threshold_down,threshold_up,apertureSize = 3)
    print(edges)
    plt.imshow(edges)
    plt.title('Canny image')
    plt.show()
    
    
    lines = []
    len_lines = min(gray.shape[0], gray.shape[1])
    while (lines is None or len(lines) < 5) and len_lines > 0.3*min(gray.shape[0], gray.shape[1]):
        print('test len', len_lines)
        lines = cv2.HoughLines(edges,1,np.pi/180,80)
        len_lines = 0.7*len_lines
    if len_lines <= 1:
        print('ABORTED')
    print(len(lines), 'lines')
    lines = [list(e[0]) for e in lines]
    direction = []
    for rho,theta in lines:
        theta = (theta + math.pi/2)%math.pi - math.pi/2
        direction.append(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite('houghlines.jpg',img)
    plt.imshow(img)
    plt.title('houghline')
    plt.show()
    # print(direction)
    my_direction = np.mean(direction)*180/np.pi
    my_std = np.std(direction)*180/np.pi
    print('Direction :', my_direction, '±', my_std)
    return my_direction, len(lines)
    
def polyfit(total_bright):
    x = list(range(len(total_bright)))
    coefs = poly.polyfit(x, total_bright, 5)
    y_base = poly.polyval(x, coefs)
    plt.figure(1)
    plt.plot(x, total_bright, 'r--')
    plt.plot(x, y_base)
    plt.title('Polyfit')
    plt.show()
    
    y = total_bright - y_base
    y = [i+128 for i in y]
    
    detection = [softmax(i) for i in y]

    return detection
    
def exacerbate(total_bright):
    x = list(range(len(total_bright)))
    
    y_base = []
    width = int(AVG_WIDTH_FURROW/2)
    for i in range(len(total_bright)):
        y_base.append(np.mean(total_bright[max(0, i-width):min(len(total_bright)-1, i+width)]))
    
    plt.figure(1)
    plt.title('Local means (exacerbate)')
    plt.plot(x, total_bright, 'r--')
    plt.plot(x, y_base)
    plt.show()
    
    y = [tot - base for tot, base in zip(total_bright, y_base)]
    y = [i+128 for i in y]
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
    
def gray_image(img, with_clahe=True):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if with_clahe:
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    return gray

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
    
def find_rotation(img):
    steps = [1, 0.1]
    theta_extremum = [0, 180] # you can reduce if you have a basic estimate
    for step in steps:
        theta_range = np.arange(theta_extremum[0], theta_extremum[1], step)
        is_first = (step is steps[0])
        best_theta = get_rotation_with_higher_variance(img, theta_range, step, is_first)
        theta_extremum = [best_theta - 2*step, best_theta + 2*step]
    print('Best', best_theta)
    return best_theta
    
def get_rotation_with_higher_variance(img, theta_range, step, is_first):
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
        plt.figure(1)
        plt.title('Fitting the gobal shape')
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

    
