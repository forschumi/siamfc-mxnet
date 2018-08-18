# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:57:16 2018

@author: pgao
"""

import mxnet as mx
import time
from mxnet import ndarray as nd
from mxnet import image
import numpy as np
import sys
import cv2
sys.path.append('../')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def tracker(siamfc, params, frame_name_list, pos_x, pos_y, target_w, target_h, ctx=mx.cpu()):
    pos_x = pos_x - 1
    pos_y = pos_y - 1
    # Load Video Information
    z = image.imread(frame_name_list[params.startFrame]).astype('float32') # H W C
#    frame_sz = z.shape # H W C
    avgChans = nd.mean(z, axis=[0, 1]) 
    nImgs = np.size(frame_name_list)
    
    context = params.contextAmount * (target_w + target_h)
    wc_z = target_w + context
    hc_z = target_h + context
    s_z = params.exemplarSize / 127 * np.sqrt(np.prod(wc_z * hc_z))
    s_x = params.instanceSize / 127 * np.sqrt(np.prod(wc_z * hc_z))
    scales = params.scaleStep ** np.linspace(np.ceil(params.numScale/2 - params.numScale), np.floor(params.numScale/2), params.numScale)
    scaledExemplar = s_z * scales
    
    z_crop_, _ = make_scale_pyramid(z, pos_x, pos_y, scaledExemplar, params.exemplarSize, avgChans, params, ctx=ctx) # B H W C
    z_crop = z_crop_[1]
    z_crop = nd.expand_dims(z_crop, axis = 0)
    z_crop = np.transpose(z_crop, axes = (0, 3, 1, 2))
    
    z_out_val = siamfc.net(z_crop.as_in_context(ctx))
    
    min_s_x = params.minSFactor * s_x
    max_s_x = params.maxSFactor * s_x
    min_s_z = params.minSFactor * s_z
    max_s_z = params.maxSFactor * s_z
    
    # cosine window to penalize large displacements   
    window_hann_1d = np.expand_dims(np.hanning(params.responseUp * params.scoreSize), axis = 0)
    window_hann_2d = np.transpose(window_hann_1d) * window_hann_1d
    window = window_hann_2d / np.sum(window_hann_2d)
    # stores tracker's output for evaluation
    print('Frame: %d' % (params.startFrame + 1))
    bboxes = np.zeros((nImgs, 4))
    bboxes[0,:] = [pos_x + 1-target_w / 2, pos_y + 1-target_h / 2, target_w, target_h]
    
    t_start = time.time()
    
    for i in range(params.startFrame + 1, nImgs):
        print('Frame: %d' % (i + 1))
        scaledInstance = s_x * scales
        x = image.imread(frame_name_list[i]).astype('float32') # H W C
        x_crops, pad_masks_x = make_scale_pyramid(x, pos_x, pos_y, scaledInstance, params.instanceSize, avgChans, params, ctx=ctx) # B H W C
        x_crops_ = np.transpose(x_crops, axes = (0, 3, 1, 2))
        x_out = siamfc.net(x_crops_.as_in_context(ctx))
        responseMaps = siamfc.match_templates(z_out_val, x_out) # B C H W        
        pos_x, pos_y, newScale = tracker_step(responseMaps, pos_x, pos_y, s_x, window, params)
        s_x = np.maximum(min_s_x, np.minimum(max_s_x, (1.0 - np.float64(params.scaleLR))* s_x + np.float64(params.scaleLR) * scaledInstance[newScale]))
        
        if params.zLR >0:
            scaledExemplar = s_z * scales
            z_crop_, _ = make_scale_pyramid(x, pos_x, pos_y, scaledExemplar, params.exemplarSize, avgChans, params, ctx=ctx) # B H W C
            z_crop = z_crop_[1]
            z_crop = nd.expand_dims(z_crop, axis = 0)
            z_crop = np.transpose(z_crop, axes = (0, 3, 1, 2))    
            z_out_val_new = siamfc.net(z_crop.as_in_context(ctx))
            z_out_val = (1 - params.zLR) * z_out_val + params.zLR * z_out_val_new
            s_z = np.maximum(min_s_z, np.minimum(max_s_z, (1 - params.scaleLR) * s_z + params.scaleLR * scaledExemplar[newScale]))
        
        scaledTarget_x, scaledTarget_y = target_w * scales, target_h * scales
        target_w = (1 - params.scaleLR) * target_w + params.scaleLR * scaledTarget_x[newScale]
        target_h = (1 - params.scaleLR) * target_h + params.scaleLR * scaledTarget_y[newScale]
        bboxes[i-params.startFrame, :] = pos_x + 1 - target_w / 2, pos_y + 1 - target_h / 2, target_w, target_h
        
		if params.visualization:
            show_frame(x.asnumpy(), bboxes[i-params.startFrame, :], 1)
		
    t_elapsed = time.time() - t_start + 1
    speed = (nImgs - 1) / t_elapsed
    
    return bboxes, speed

def make_scale_pyramid(z, pos_x, pos_y, in_side_scaled, out_side, avgChans, params, ctx=mx.cpu()):
    n = np.size(in_side_scaled)
    in_side_scaled = np.round(in_side_scaled)
    pyramid_ = nd.zeros((out_side, out_side, 3, n), ctx=ctx) # H W C B
    pad_masks_x = np.zeros((out_side, out_side, 1, n), dtype=bool)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    search_side = np.round(out_side * max_target_side / min_target_side)
    search_region, _ = get_subwindow_tracking(z, pos_x, pos_y, search_side, max_target_side, avgChans, ctx)
    pyramid = nd.transpose(pyramid_, axes=(3,0,1,2)) # B H W C
    for s in range(n):
        target_side = np.round(out_side * in_side_scaled[s] / min_target_side)
        pyramid[s], _ = get_subwindow_tracking(search_region, (1+search_side)/2 -1, (1+search_side)/2 -1, np.float64(out_side), target_side, avgChans, ctx)
    return pyramid, pad_masks_x

def get_subwindow_tracking(z, pos_x, pos_y, model_sz, original_sz, avgChans, ctx=mx.cpu()):
    if original_sz is None:
        original_sz = model_sz
    sz = original_sz
    im_sz = np.shape(z)
    cen = (sz - 1) / 2
    
    context_xmin = np.floor(pos_x - cen)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos_y - cen)
    context_ymax = context_ymin + sz - 1
    
    left_pad = nd.maximum(0, 1 - context_xmin)
    top_pad = nd.maximum(0, 1 - context_ymin)
    right_pad = nd.maximum(0, context_xmax - im_sz[1])
    bottom_pad = nd.maximum(0, context_ymax - im_sz[0])

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;
    
    paddings = [0, 0, 0, 0, int(top_pad), int(bottom_pad), int(left_pad), int(right_pad)]
    if avgChans is not None:
        im_padded_ = z - avgChans
    im_padded_ = nd.expand_dims(im_padded_, axis = 0) # B H W C
    im_padded_ = nd.transpose(im_padded_, axes=(0,3,1,2)) # B C H W
    im_padded_ = nd.pad(im_padded_, pad_width=paddings, mode='constant')
    im_padded_ = nd.transpose(im_padded_, axes=(0,2,3,1)) # B H W C
    if avgChans is not None:
        im_padded_ = im_padded_ + avgChans
    im_padded = im_padded_[0]
    im_patch_original = im_padded[int(context_ymin - 1) : int(context_ymax), int(context_xmin - 1) : int(context_xmax), :]
    if int(model_sz) != int(original_sz):
        sz_dst_w = np.round(im_patch_original.shape[1] / original_sz * model_sz)
        sz_dst_h = np.round(im_patch_original.shape[0] / original_sz * model_sz)
        im_patch = image.fixed_crop(im_patch_original,
                                    x0 = 0,
                                    y0 = 0,
                                    w = im_patch_original.shape[1],
                                    h = im_patch_original.shape[0],
                                    size = [int(sz_dst_w), int(sz_dst_h)],
                                    interp = 1
                                    )
        if im_patch.shape[0] != model_sz:
            im_patch = image.fixed_crop(im_patch_original,
                                        x0 = 0,
                                        y0 = 0,
                                        w = im_patch_original.shape[1],
                                        h = im_patch_original.shape[0],
                                        size = [int(model_sz), int(model_sz)],
                                        interp = 1
                                        )
    else:
        im_patch = im_patch_original
    return im_patch, im_patch_original
  
def tracker_step(responseMaps, pos_x, pos_y, s_x, window, params):
    currentScaleID = np.ceil(params.numScale / 2 - 1)
    bestScale = currentScaleID
    bestPeak = np.float("-inf")
    responseMapsUP = cv2.resize(responseMaps.asnumpy(),
                                (0, 0),
                                fx = params.responseUp,
                                fy = params.responseUp,
                                interpolation = cv2.INTER_CUBIC
                                )
    responseMapsUP = np.transpose(responseMapsUP, axes = (2, 0, 1))
    for s in range(params.numScale):
        thisResponse = responseMapsUP[s]
        if s != currentScaleID:
            thisResponse = thisResponse * params.scalePenalty
        thisPeak = np.max(thisResponse)
        if thisPeak > bestPeak:
            bestPeak = thisPeak
            bestScale = s
    responseMap = responseMapsUP[bestScale]
    responseMap = responseMap - np.min(responseMap[:,:])
    responseMap = responseMap / np.sum(responseMap[:,:])
    response_final = (1 - params.wInfluence) * responseMap + params.wInfluence * window
    [r_max, c_max] = np.unravel_index(np.argmax(response_final), np.shape(response_final))
    if r_max is None:
        r_max = np.ceil(params.scoreSize / 2 - 1)
    if c_max is None:
        c_max = np.ceil(params.scoreSize / 2 - 1)
    disp_instanceFinal_r = r_max - (params.scoreSize * params.responseUp - 1) / 2 + 1
    disp_instanceFinal_c = c_max - (params.scoreSize * params.responseUp - 1) / 2 + 1
    disp_instanceInput_r = disp_instanceFinal_r * params.totalStride / params.responseUp
    disp_instanceInput_c = disp_instanceFinal_c * params.totalStride / params.responseUp
    disp_instanceFrame_r = disp_instanceInput_r * s_x / params.instanceSize
    disp_instanceFrame_c = disp_instanceInput_c * s_x / params.instanceSize
    new_pos_x, new_pos_y = pos_x + disp_instanceFrame_c, pos_y + disp_instanceFrame_r
    
    return new_pos_x, new_pos_y, bestScale

def show_frame(frame, bbox, fig_n):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()	
	
def _try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx  