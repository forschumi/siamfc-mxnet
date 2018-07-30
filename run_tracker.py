# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:54:35 2018

@author: pgao
"""

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import image
import numpy as np
import sys
sys.path.append('../')
import os
from hyperparams.params import paramsInitial
from tracking.tracker import tracker

def main():
    ctx = _try_gpu()
    params = paramsInitial()
    final_score_sz = params.response_up * (params.score_sz - 1) + 1
    # iterate through all videos of evaluation.dataset
    if params.video == 'all':
        dataset_folder = os.path.join(params.root_dataset)
        videos_list = [v for v in os.listdir(dataset_folder)]
        videos_list.sort()
        nv = np.size(videos_list)
        speed = np.zeros(nv)
        precisions = np.zeros(nv)
        precisions_auc = np.zeros(nv)
        ious = np.zeros(nv)
        lengths = np.zeros(nv)
        for i in range(nv):
            gt, frame_name_list, frame_sz, n_frames = _load_video_info(videos_list[i], params)
            gt_ = gt[params.start_frame:, :]
            frame_name_list_ = frame_name_list[params.start_frame:]
            pos_x, pos_y, target_w, target_h = _region_to_bbox(gt_[0], center = True)
            bboxes, speed = tracker(params, frame_name_list_, pos_x, pos_y, target_w, target_h,
                                    final_score_sz, params.start_frame, ctx=ctx)
            lengths[i], precisions[i], precisions_auc[i], ious[i] = _compile_results(gt_, bboxes, params.dist_threshold)
            print(str(i) + ' -- ' + videos_list[i] + \
                ' -- Precision ' + "(%d px)" % params.dist_threshold + ': ' + "%.2f" % precisions[i] +\
                ' -- Precisions AUC: ' + "%.2f" % precisions_auc[i] + \
                ' -- IOU: ' + "%.2f" % ious[i] + \
                ' -- Speed: ' + "%.2f" % speed[i] + ' --')
            print()

        tot_frames = np.sum(lengths)
        mean_precision = np.sum(precisions * lengths) / tot_frames
        mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
        mean_iou = np.sum(ious * lengths) / tot_frames
        mean_speed = np.sum(speed * lengths) / tot_frames
        print('-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --')
        print(' -- Precision ' + "(%d px)" % params.dist_threshold + ': ' + "%.2f" % mean_precision +\
              ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc +\
              ' -- IOU: ' + "%.2f" % mean_iou +\
              ' -- Speed: ' + "%.2f" % mean_speed + ' --')
        print()

    else:
        dataset_folder = os.path.join(params.root_dataset)
        nv = np.size(params.video)
        speed = np.zeros(nv)
        precisions = np.zeros(nv)
        precisions_auc = np.zeros(nv)
        ious = np.zeros(nv)
        lengths = np.zeros(nv)
        for i in range(nv):
            gt, frame_name_list, frame_sz, n_frames = _load_video_info(params.video[i], params)
            pos_x, pos_y, target_w, target_h = _region_to_bbox(gt[params.start_frame], center = True)
            bboxes, speed = tracker(params, frame_name_list, pos_x, pos_y, target_w, target_h,
                                    final_score_sz, params.start_frame, ctx=ctx)
            _, precision, precision_auc, iou = _compile_results(gt, bboxes, params.dist_threshold)
            print(str(i) + ' -- ' + params.video[i] + \
                  ' -- Precision ' + "(%d px)" % params.dist_threshold + ': ' + "%.2f" % precisions[i] +\
                  ' -- Precision AUC: ' + "%.2f" % precision_auc[i] + \
                  ' -- IOU: ' + "%.2f" % ious[i] + \
                  ' -- Speed: ' + "%.2f" % speed[i] + ' --')
            print()


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = _region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)    

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


def _load_video_info(video, params):
    video_folder = os.path.join(params.root_dataset, video)
    frame_folder = os.path.join(video_folder, 'img')
    frame_name_list = [f for f in os.listdir(frame_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(frame_folder, s) for s in frame_name_list]
    frame_name_list.sort()
    img = image.imdecode(open(frame_name_list[0], 'rb').read())
    frame_sz = np.array(img.shape).astype('int32')
    frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth_rect.txt')
    gt = np.array(np.genfromtxt(gt_file, delimiter=','))
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames

def _region_to_bbox(region, center=False):

    n = len(region)
    assert n==4 or n==8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n==4:
        return _rect(region, center)
    else:
        return _poly(region, center)

# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
        return cx, cy, w, h
    else:
        #region[0] -= 1
        #region[1] -= 1
        return region


def _poly(region, center):
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

    if center:
        return cx, cy, w, h
    else:
        return cx-w/2, cy-h/2, w, h

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def _try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx  

if __name__ == '__main__':
    sys.exit(main())
    