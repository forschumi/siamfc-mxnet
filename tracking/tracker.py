# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:57:16 2018

@author: pgao
"""

import mxnet as mx
import time
from mxnet import ndarray as nd
from mxnet import init, image
from mxnet.gluon import utils,nn
import numpy as np
import scipy.io as sio
import os
import sys
sys.path.append('../')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from hyperparams.params import paramsInitial
import nets.siamese as siamese
from tracking.visualization import show_frame

_stats_path = './nets/cfnet_ILSVRC2015.stats.mat'

def tracker(params, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, start_frame, ctx=mx.cpu()):
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames,4))
    bboxes[0,:] = [pos_x-target_w/2, pos_y-target_h/2, target_w, target_h]    
    scale_factors = params.scale_step ** np.linspace(np.ceil(params.scale_num/2 - params.scale_num), np.floor(params.scale_num/2), params.scale_num)
    # cosine window to penalize large displacements   
    window_hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    window_hann_2d = np.transpose(window_hann_1d) * window_hann_1d
    window = window_hann_2d / np.sum(window_hann_2d)
    
    context = params.context * (target_w + target_h)
    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
    x_sz = float(params.search_sz) / params.exemplar_sz * z_sz
    # Load Video Information
    z = image.imread(frame_name_list[0]).astype('float32') # H W C
    frame_sz = z.shape # H W C
    if params.pad_with_image_mean:
        avg_chan = nd.mean(z, axis=[0, 1])
    else:
        avg_chan = None
    
    siamfc = siamese.SiamFC()
    siamfc.collect_params().initialize(ctx=ctx)
    frame_padded_z, npad_z = pad_frame(z, frame_sz, pos_x, pos_y, z_sz, avg_chan)
    z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, params.exemplar_sz)
    z_crops = nd.transpose(z_crops, (0,3,2,1)) # B C H W
    templates_z = siamfc.net(z_crops.as_in_context(ctx))
    new_templates_z = templates_z.copy()
    
    t_start = time.time()
    
    for i in range(1, num_frames):
        print('Frame: %d' % i)
        scaled_exemplar = z_sz * scale_factors
        scaled_search_area = x_sz * scale_factors
        scaled_target_w = target_w * scale_factors
        scaled_target_h = target_h * scale_factors
        
        x = image.imread(frame_name_list[i]).astype('float32') # H W C
        frame_padded_x, npad_x = pad_frame(x, frame_sz, pos_x, pos_y, x_sz, avg_chan)
        x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, *scaled_search_area, params.search_sz)
        x_crops = mx.nd.transpose(x_crops, (0,3,2,1)) # 
        templates_x = siamfc.net(x_crops.as_in_context(ctx))        
        scores = siamfc.match_templates(templates_z, templates_x)
        scores_up = nd.zeros(shape=[3, final_score_sz, final_score_sz], ctx = ctx)
        for j in range(3):
            scores_up[j] = image.resize_short(src = scores[j, 0].expand_dims(axis = 2).as_in_context(mx.cpu()), 
                     size = final_score_sz, interp = 1)[:,:,0]
        scores_up[0, :, :] = params.scale_penalty * scores_up[0, :, :]
        scores_up[2, :, :] = params.scale_penalty * scores_up[2, :, :]
        new_scale_id = np.int(np.argmax(np.amax(scores_up.asnumpy(), axis=(1, 2))))
        x_sz = (1 - params.scale_lr) * x_sz + params.scale_lr * scaled_search_area[new_scale_id]
        target_w = (1 - params.scale_lr) * target_w + params.scale_lr * scaled_target_w[new_scale_id]
        target_h = (1 - params.scale_lr) * target_h + params.scale_lr * scaled_target_h[new_scale_id]
        score_ = scores_up[new_scale_id, :, :]
        score_ = score_ - nd.min(score_)
        score_ = score_ / nd.sum(score_)
        score_ = score_.asnumpy()
        score_ = (1 - params.window_influence) * score_ + params.window_influence * window
        pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, params.tot_stride, params.search_sz, params.response_up, x_sz)
        bboxes[i, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
        if params.z_lr>0:
            if params.pad_with_image_mean:
                avg_chan = nd.mean(x, axis=[0, 1])
            else:
                avg_chan = None
            frame_padded_z, npad_z = pad_frame(x, frame_sz, pos_x, pos_y, z_sz, avg_chan)
            z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, params.exemplar_sz)
            z_crops = nd.transpose(z_crops, (0,3,2,1))
            new_templates_z = siamfc.net(z_crops.as_in_context(ctx))
            templates_z = (1 - params.z_lr) * templates_z + params.z_lr * new_templates_z
        if params.visualization:
                show_frame(x, bboxes[i,:], 1) 
        z_sz = (1 - params.scale_lr) * z_sz + params.scale_lr * scaled_exemplar[new_scale_id]
                
    t_elapsed = time.time() - t_start + 1
    speed = (num_frames - 1) / t_elapsed
    
    return bboxes, speed

def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    pos_x = nd.array([pos_x])
    pos_y = nd.array([pos_y])
    c = np.round(patch_sz) / 2
    xleft_pad = nd.maximum(0, -nd.round(pos_x - c).astype('int32'))
    ytop_pad = nd.maximum(0, -nd.round(pos_y - c).astype('int32'))
    xright_pad = nd.maximum(0, nd.round(pos_x + c).astype('int32') - frame_sz[1])
    ybottom_pad = nd.maximum(0, nd.round(pos_y + c).astype('int32') - frame_sz[0])
    npad = nd.max(nd.concat(xleft_pad, ytop_pad, xright_pad, ybottom_pad, dim = 0)).asscalar()
    paddings = [0, 0, 0, 0, npad, npad, npad, npad]
    im_padded = nd.expand_dims(im, axis = 0) # B H W C
    if avg_chan is not None:
        im_padded = im_padded - avg_chan
    im_padded = np.transpose(im_padded, axes=(0,3,1,2)) # B C H W
    im_padded = nd.pad(im_padded, pad_width=paddings, mode='constant')
    im_padded = np.transpose(im_padded, axes=(0,2,3,1)) # B H W C
    if avg_chan is not None:
        im_padded = im_padded + avg_chan
    return im_padded[0], npad # H W C

def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    pos_x = nd.array([pos_x])
    pos_y = nd.array([pos_y])
    c = np.round(sz_src + 1) / 2
    # get top-right corner of bbox and consider padding
    tl_x = nd.maximum(0, npad + nd.floor(pos_x - c))
    tl_y = nd.maximum(0, npad + nd.floor(pos_y - c))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = nd.maximum(0, nd.round(pos_x + c) - nd.round(pos_x - c))
    height = nd.maximum(0, nd.round(pos_y + c) - nd.round(pos_y - c))
    crop = image.fixed_crop(im,
                            x0 = int(tl_x.asscalar()),
                            y0 = int(tl_y.asscalar()),
                            w = int(width.asscalar()),
                            h = int(height.asscalar()),
                            size = [sz_dst, sz_dst],
                            interp = 1
                            )
    # crops = nd.stack([crop, crop, crop])
    crops = nd.expand_dims(crop, axis=0)
    crops = nd.stack(crop, crop, crop, axis = 0) # B W H C
    return crops

def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    # take center of the biggest scaled source patch
    pos_x = nd.array([pos_x])
    pos_y = nd.array([pos_y])
    sz_src0, sz_src1, sz_src2 = nd.array([sz_src0, sz_src1, sz_src2])
    c = nd.round(sz_src2 + 1) / 2
    # get top-right corner of bbox and consider padding
    tr_x = nd.maximum(0, npad + nd.floor(pos_x - c))
    tr_y = nd.maximum(0, npad + nd.floor(pos_y - c))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = nd.maximum(0, nd.round(pos_x + c) - nd.round(pos_x - c))
    height = nd.maximum(0, nd.round(pos_y + c) - nd.round(pos_y - c))
    search_area = image.fixed_crop(im,
                                   int(tr_x.asscalar()),
                                   int(tr_y.asscalar()),                                   
                                   int(width.asscalar()),
                                   int(height.asscalar()),                                   
                                   )

    offset_s0 = np.round((sz_src2 - sz_src0) / 2)
    offset_s1 = np.round((sz_src2 - sz_src1) / 2)

    crop_s0 = image.fixed_crop(search_area,
                               int(offset_s0.asscalar()),
                               int(offset_s0.asscalar()),
                               int(sz_src0.asscalar()),
                               int(sz_src0.asscalar()),
                               size = [sz_dst, sz_dst],
                               interp = 1
                               )

    crop_s1 = image.fixed_crop(search_area,
                               int(offset_s1.asscalar()),
                               int(offset_s1.asscalar()),
                               int(sz_src1.asscalar()),
                               int(sz_src1.asscalar()),
                               size = [sz_dst, sz_dst],
                               interp = 1
                               )

    crop_s2 = image.fixed_crop(search_area, 
                               0,
                               0, 
                               search_area.shape[1],
                               search_area.shape[0],
                               size = [sz_dst, sz_dst],
                               interp = 1
                               )
    crops = nd.stack(crop_s0, crop_s1, crop_s2)
    return crops

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y
  
def _try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx  