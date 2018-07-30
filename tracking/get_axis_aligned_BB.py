# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:07:51 2018

@author: pgao
"""

import numpy as np

def get_axis_aligned_BB(region):

    n = len(region)
    assert n == 4 or n == 8, ('GroundTruth must have 4 or 8 entries.')

    if n == 4:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w/2
        cy = y + h/2    
    else:
        cx = np.mean(region[::2])
        cy = np.mean(region[1::2])
        x1 = np.min(region[::2])
        x2 = np.max(region[::2])
        y1 = np.min(region[1::2])
        y2 = np.max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1/A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        
    return cx, cy, w, h
