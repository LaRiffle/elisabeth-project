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
from furrows_detector import get_furrows_gap

paths = [24,25,26,27]
gaps_list = []
for idx in paths:
    path = 'sillons_cadres/sillon' + str(idx) + '.png'
    print(path)
    gaps = get_furrows_gap(path)
    gaps_list.append(gaps)
    
gaps_merged = gaps_list[0]
for i in range(1, len(gaps_list)):
    gaps_merged = match_lists(gaps_merged, gaps_list[i])
    
    
def match_lists(l1, l2, verb=''):
    """
        Translate one list compared to another and find the best match
        l1 = [1,3,5,7,9]
        l2 = [3,4,7]
        Best match :
        [1, 3, 5, 7, 9]
        ...[3, 4, 7]...
    """
    # match must have a minimum length
    min_length_match = 5
    
    if len(l1) >= len(l2):
        big_list, short_list = l1, l2
    else:
        big_list, short_list = l2, l1
        
    best_score = 2**63 - 1
    best_translation = -1
    # translate one list % another and calculate a matching score
    for index in range(1, len(big_list)+len(short_list)):
        if 'v' in verb:
            print('#',index)
        if index < len(short_list):
            tmp_short_list = short_list[len(short_list) - index:]
        elif index > len(big_list):
            tmp_short_list = short_list[:len(big_list) - index]
        else:
            tmp_short_list = short_list
        if index > len(short_list):
            tmp_big_list = big_list[index - len(short_list):index]
        else:
            tmp_big_list = big_list[:index]

        if 'v' in verb:
            print(tmp_big_list)
            print(tmp_short_list)
            
        score = 0
        for i1, i2 in zip(tmp_big_list, tmp_short_list):
            score += (i2 - i1)**2
        score /= len(tmp_short_list)
        if len(tmp_short_list) >= min_length_match:
            if score < best_score:
                best_score = score
                best_translation = index
            if 'v' in verb:
                print(score)
        else:
            if 'v' in verb:
                print('(', score, ')')
    offset = len(short_list)
    translation = best_translation - offset
    if 'v' in verb:
        print('# best', best_translation)
        print('translation', translation)
    
    if translation >= 0:
        base_list, additional_list = big_list, short_list
    else:
        base_list, additional_list = short_list, big_list
        
    translation = abs(translation)
    if 'v' in verb:
        print(base_list)
        print(['']*translation + additional_list)
    
    matched_list_size = max(len(base_list), translation + len(additional_list))
    # determine on which indices the two lists overlap
    indices_to_merge = {'start': translation,
                        'end': min(len(base_list), translation + len(additional_list))
                        }
    # beginning with one list
    matched_list = base_list[:translation]
    # middle with merge of the two overlapping lists
    for i in range(indices_to_merge['start'], indices_to_merge['end']):
        nb = (base_list[i] + additional_list[i - translation])/2
        matched_list.append(nb)
    # tail of the list
    if matched_list_size > len(base_list):
        matched_list += additional_list[len(base_list) - translation:]
    else:
        matched_list += base_list[translation + len(additional_list):]

    matched_list = list(map(float, matched_list))
    print(matched_list)
    return matched_list
    
l2 = [11, 14, 14, 14, 14, 14, 38, 22, 16, 16, 17, 17, 15, 17, 16, 16, 14]
l1 = [12, 14, 14, 13, 13, 14, 13, 13, 14, 14, 14, 14, 15, 38, 23, 17, 16, 12]

#match_lists(l1, l2, verb='')