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
import scipy.io as sio
from hyperparams.params import paramsInitial
import nets.siamese as siamese
from tracking.tracker import tracker

def main():
    ctx = _try_gpu()
    params = paramsInitial()
    siamfc = siamese.SiamFC()
    siamfc.collect_params().initialize(ctx=ctx)
    siamfc.net.load_params('./nets/siamfc_net.params')
    siamfc.bn_final.load_params('./nets/siamfc_bn.params')
    # iterate through all videos of evaluation.dataset
    if params.all is True:
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
            if videos_list[i] == "David":
                startFrame = 300
            else:
                startFrame = params.startFrame + 1
            pos_x, pos_y, target_w, target_h = _region_to_bbox(gt[params.startFrame], center = True)
            bboxes, speed_ = tracker(siamfc, params, frame_name_list, pos_x, pos_y, target_w, target_h, ctx=ctx)
            lengths[i], precisions[i], precisions_auc[i], ious[i] = _compile_results(gt, bboxes, params.dist_threshold, params.iou_threshold)
            speed[i] = speed_
            print(str(i+1) + ' -- ' + videos_list[i])
            print(' -- Precision (%d px): %.2f' % (params.dist_threshold, precisions[i]))            
            print(' -- IOU: %.2f' % ious[i])
            print(' -- Speed: %.2f' % speed_)
            print(' -- Precisions AUC (%.1f): %.2f' % (params.iou_threshold, precisions_auc[i]))
            print(' --')
            print()
            results_ =  {'type':'rect', 'res': bboxes, 'fps': speed_, \
                         'len': len(frame_name_list), 'annoBegin': startFrame, \
                         'startFrame': startFrame}
            results = np.zeros((1,), dtype=np.object)
            results[0] = results_
            sio.savemat('results/'+videos_list[i]+'_SiamFC_MXNet.mat', {'results' : results})
        tot_frames = np.sum(lengths)
        mean_precision = np.sum(precisions * lengths) / tot_frames
        mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
        mean_iou = np.sum(ious * lengths) / tot_frames
        mean_speed = np.sum(speed * lengths) / tot_frames
        print(' -- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --')
        print(' -- Precision (%d px): %.2f' % (params.dist_threshold, mean_precision)) 
        print(' -- IOU: %.2f' % mean_iou)
        print(' -- Speed: %.2f' % mean_speed)
        print(' -- Precisions AUC (%.1f): %.2f' % (params.iou_threshold, mean_precision_auc))
        print(' --')
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
            if params.video[i] == "David":
                startFrame = 300
            else:
                startFrame = params.startFrame
            pos_x, pos_y, target_w, target_h = _region_to_bbox(gt[params.startFrame], center = True)
            bboxes, speed = tracker(siamfc, params, frame_name_list, pos_x, pos_y, target_w, target_h, ctx=ctx)
            _, precision, precision_auc, iou = _compile_results(gt, bboxes, params.dist_threshold, params.iou_threshold)
            print(str(i) + ' -- ' + params.video[i])
            print('  -- Precision (%dpx): %.2f' % (params.dist_threshold, precision))
            print('  -- IOU: %.2f' % iou)
            print('  -- Speed: %.2f' % speed)
            print('  -- Precision AUC (%.1f): %.2f' % (params.iou_threshold, precision_auc))
            print()
            results_ =  {'type':'rect', 'res': bboxes, 'fps': speed, \
                         'len': len(frame_name_list), 'annoBegin': startFrame, \
                         'startFrame': startFrame}
            results = np.zeros((1,), dtype=np.object)
            results[0] = results_
            sio.savemat('results/'+params.video[i]+'_SiamFC_MXNet.mat', {'results' : results})


def _compile_results(gt, bboxes, dist_threshold, iou_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_iou = 20
    iou_ths = np.zeros(n_iou)

    for i in range(l):
        gt4[i, :] = _region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100   

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100
    iou_thresholds = np.linspace(0, 1, n_iou+1)
    for i in range(n_iou):
        iou_ths[i] = sum(new_ious > iou_thresholds[i])/np.size(new_ious)
    precision_auc = np.trapz(iou_ths) / n_iou * 100

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
    #assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

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
    