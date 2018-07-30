# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:32:11 2018

@author: pgao
"""

from mxnet import image
import os
import numpy as np
from tracking.get_axis_aligned_BB import get_axis_aligned_BB

def load_video_info(video_base_path, video):
    video_path = os.path.join(video_base_path, video)
    frame_path = os.path.join(video_base_path, video,'img')
    
    imgs = [f for f in os.listdir(video_path + '/img/') if f.endswith(".jpg")]
    imgs = [os.path.join(frame_path, s) for s in imgs]
    imgs.sort()
    img = image.imdecode(open(imgs[0], 'rb').read())
    imgSize = np.array(img.shape).astype('int32')
    imgSize[1], imgSize[0] = imgSize[0], imgSize[1]

    groundtruths = os.path.join(video_path, 'groundtruth_rect.txt')
    groundtruth = np.array(np.genfromtxt(groundtruths, delimiter=','))
    cx, cy, cw, ch = get_axis_aligned_BB(groundtruth[0])
    pos = np.array([cy, cx])
    target_sz = np.array([ch, cw])
    
    imgFrames = len(imgs)
    assert imgFrames == len(groundtruth), 'Number of frames and number of GT lines should be equal.'

    return imgs, pos, target_sz, imgFrames, imgSize
