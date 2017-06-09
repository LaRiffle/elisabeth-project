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

path = 'sillons_cadres/sillon3.png'

def get_canny_configuration(gray):
    best_score = 0
    threshold = {'down':0, 'up':0}
    for down in np.arange(5, 251, 10):
        for width in np.arange(5, 201, 10):
            threshold_down = down
            threshold_up = min(down + width, 255)
            edges = cv2.Canny(gray,threshold_down,threshold_up,apertureSize = 3)
            lines = cv2.HoughLines(edges,1,np.pi/180,80)
            if lines is None:
                lines = []
            lines = [list(e[0]) for e in lines]
            direction = []
            for rho,theta in lines:
                direction.append(theta)
            my_direction = np.mean(direction)*180/np.pi
            my_std = np.std(direction)*180/np.pi
            if len(lines) < 3:
                score = 0
            else:
                my_std += 0.1
                score = min(len(lines), 50)/my_std
            #print('conf', threshold_down, threshold_up, len(lines), 'lines', 'dir :', my_direction, '±', my_std, 'score', score)
            if score > best_score:
                best_score = score
                threshold['down'] = threshold_down
                threshold['up'] = threshold_up
    print('Recommended', threshold, best_score)
    return threshold
            

def canny_direction(gray,threshold_down, threshold_up):
    edges = cv2.Canny(gray,threshold_down,threshold_up,apertureSize = 3)
    plt.imshow(edges)
    plt.title('Canny image')
    plt.show()
    
    
    lines = []
    len_lines = min(img.shape[0], img.shape[0])
    while (lines is None or len(lines) < 5) and len_lines > 0.3*min(img.shape[0], img.shape[0]):
        print('test len', len_lines)
        lines = cv2.HoughLines(edges,1,np.pi/180,80)
        len_lines = 0.7*len_lines
    if len_lines <= 1:
        print('ABORTED')
    print(len(lines), 'lines')
    lines = [list(e[0]) for e in lines]
    direction = []
    for rho,theta in lines:
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
    #print(direction)
    my_direction = np.mean(direction)*180/np.pi
    my_std = np.std(direction)*180/np.pi
    print('Direction :', my_direction, '±', my_std)
    return my_direction


# I. Define rough orientation
print('I. Define rough orientation')
img = cv2.imread(path)
#gray = cv2.equalizeHist(gray)
plt.imshow(img)
plt.title('Natural image')
plt.show()

print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
plt.title('Gray image - clahe')
plt.imshow(gray)
plt.show()


threshold = get_canny_configuration(gray)
theta = canny_direction(gray, threshold['down'], threshold['up'])


# II. Precise orientation
print(' II. Precise orientation')

img = cv2.imread(path)
# rotation
#theta = 51
rows,cols,rgb = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
dst = cv2.warpAffine(img,M,(cols,rows))
plt.imshow(dst)
plt.title('Rotated image '+str(theta)+'°')
plt.show()

# crop the image and substitute
rows,cols,rgb = dst.shape
crop_img = dst[int(0.25*rows):int(0.75*rows), int(0.25*cols):int(0.75*cols)] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
plt.imshow(crop_img)
plt.title('Croped image')
plt.show()

img = crop_img

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = clahe.apply(gray)
#gray = cv2.equalizeHist(gray)
plt.title('Gray image - clahe')
plt.imshow(gray)
plt.show()

threshold = get_canny_configuration(gray)
theta_cor = canny_direction(gray, threshold['down'], threshold['up'])

# III. Count sillons
print(' III. Count sillons')

img = cv2.imread(path)
# rotation
theta = theta + theta_cor
rows,cols,rgb = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
dst = cv2.warpAffine(img,M,(cols,rows))
plt.imshow(dst)
plt.title('Rotated image '+str(theta)+'°')
plt.show()

# crop the image and substitute
rows,cols,rgb = dst.shape
crop_img = dst[int(0.25*rows):int(0.75*rows), int(0.25*cols):int(0.75*cols)] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
plt.imshow(crop_img)
plt.title('Croped image')
plt.show()

img = crop_img

# preprare gray for canny : good
print(img.shape)
h, w, _ = img.shape
gray = []
for i in range(h):
    gray.append([np.mean(img[:, i]) for i in range(w)])
    
gray = np.uint8(np.array(gray))
gray = clahe.apply(gray)
print(gray.shape)

plt.title('Gray image - clahe')
plt.imshow(gray)
plt.show()

threshold = get_canny_configuration(gray)
threshold = {'down':30, 'up': 50}
canny_direction(gray, threshold['down'], threshold['up'])


#try to do wihtout canny : bad
'''mean = np.mean(np.linalg.norm(img, axis=2))
edges = np.uint8(np.array([ [ 255 if np.linalg.norm(pix)>(mean) and np.linalg.norm(pix)<(mean+10) else 0 for pix in line ] for line in img]))
print(edges)
print(edges.shape)
plt.imshow(edges)
plt.show()
print(type(edges))'''

'''
# preprare gray for canny : good
print(img.shape)
h, w, _ = img.shape
gray = []
for i in range(h):
    gray.append([np.mean(img[:, i]) for i in range(w)])
    
gray = np.uint8(np.array(gray))
print(gray.shape)

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray)

plt.show()
print(type(gray))
'''

