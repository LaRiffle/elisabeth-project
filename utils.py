#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math


def get_canny_configuration(gray):
    best_score = 0
    threshold = {'down':0, 'up':0}
    for down in np.arange(0, 241, 10):
        for width in np.arange(20, 221, 10):
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
    
        
def part(title):
    print()
    print(title)
    print()
    
