# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 08:23:56 2018

@author: pgao
"""

from mxnet import ndarray as nd
from mxnet import init
from mxnet.gluon import nn
import numpy as np
import scipy.io as sio

# the follow parameters *have to* reflect the design of the network to be imported
_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_kernel_sz = np.array([11,5,3,3,3])
_in_c = np.array([3,48,256,192,192])
_out_c = np.array([96,256,384,384,32])
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags must have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)

class SiamInit(init.Initializer):
    def __init__(self, w):
        super(SiamInit, self).__init__()
        self._verbose = True
        self.w = nd.array(w)
    def _init_weight(self, _, arr):
        nd.reshape(self.w, out = arr, shape = arr.shape)     

class SiamFC(nn.Block):
    def __init__(self, verbose=False, **kwargs):
        super(SiamFC, self).__init__(**kwargs)
        self.verbose = verbose
        if _bnorm_adjust:
            self.bn_final = nn.BatchNorm(use_global_stats=True)
        self.net = nn.Sequential()
        for i in range(_num_layers):
            print('> Layer '+str(i+1))
            # set up conv "block" with bnorm and activation
            _set_convolutional(self.net, _out_c[i], _in_c[i], _kernel_sz[i], _conv_stride[i],
                               filtergroup = _filtergroup_yn[i],
                               batchnorm = _bnorm_yn[i],
                               activation = _relu_yn[i]
                               )
            if _pool_stride[i]>0:
                self.net.add(nn.MaxPool2D(pool_size = _pool_sz, strides = _pool_stride[i]))
     
    def forward(self, z, x):
        net_z = self.net(z)
        net_x = self.net(x)
        print(net_z.shape)
        print(net_x.shape)
        out = self.match_templates(self, net_z, net_x)
        return out

    def match_templates(self, net_z, net_x):
        # B C H W
        Bz, Cz, Hz, Wz = net_z.shape
        Bx, Cx, Hx, Wx = net_x.shape
        net_final_ = nd.Convolution(data = net_x, 
                                   weight = net_z, 
                                   num_filter = 1,
                                   kernel = [Hz, Wz],
                                   no_bias = True
                                   )
        net_final_ = self.bn_final(net_final_)
        net_final_ = np.transpose(net_final_, axes = (1, 2, 3, 0))
        net_final = net_final_[0]
        return net_final

def _import_from_matconvnet(net_path):
    mat = sio.loadmat(net_path)
    nets = mat.get('net')
    params = nets['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_values = params['value'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values_list_ = [params_values[p] for p in range(params_values.size)]
    params_values_list=[]
    for i, p in enumerate(params_values_list_):
        if len(p.shape) < 4:
            params_values_list.append(p)
        else:
            params_values_list.append(np.transpose(p, axes=(3,2,0,1))) # FN FC FH FW
    return params_names_list, params_values_list

def _find_params(x, params):                                                                         
    matching = [s for s in params if x in s]                                                         
    assert len(matching) == 1, ('Ambiguous param name found')                                          
    return matching 

def _set_convolutional(net, out_c, in_c, kernel, stride, 
                       filtergroup = False, batchnorm = True, 
                       activation = True, reuse = False):
        if filtergroup:
            net.add(nn.Conv2D(channels = out_c,
                              in_channels = in_c * 2, 
                              kernel_size = kernel,
                              strides = stride, 
                              groups = 2,
                             ))

        else:
            net.add(nn.Conv2D(channels = out_c, 
                              in_channels = in_c, 
                              kernel_size = kernel,
                              strides = stride, 
                             ))
            
        if batchnorm:
            net.add(nn.BatchNorm(use_global_stats=True))

        if activation:
            net.add(nn.Activation('relu'))
