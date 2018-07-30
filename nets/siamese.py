# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 08:23:56 2018

@author: pgao
"""

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import init
from mxnet.gluon import utils,nn
import numpy as np
import scipy.io as sio

net_path = './nets/baseline-conv5_e55.mat'
net_path_gray = './nets/baseline-conv5_gray_e100.mat'
# the follow parameters *have to* reflect the design of the network to be imported
_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
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
        self.params_names_list, self.params_values_list = _import_from_matconvnet(net_path)
        if _bnorm_adjust:
            bn_beta = self.params_values_list[self.params_names_list.index('fin_adjust_bnb')]
            bn_gamma = self.params_values_list[self.params_names_list.index('fin_adjust_bnm')]
            bn_moments = self.params_values_list[self.params_names_list.index('fin_adjust_bnx')]
            bn_moving_mean = bn_moments[:, 0]
            bn_moving_variance = bn_moments[:, 1] ** 2
            self.bn_final = nn.BatchNorm(beta_initializer = SiamInit(bn_beta),
                                         gamma_initializer = SiamInit(bn_gamma),
                                         running_mean_initializer = SiamInit(bn_moving_mean),
                                         running_variance_initializer = SiamInit(bn_moving_variance),
                                         in_channels = 1,
                                         use_global_stats = True
                                         )
        self.net = nn.Sequential()
        for i in range(_num_layers):
            print('> Layer '+str(i+1))
            # conv
            conv_W_name = _find_params('conv'+str(i+1)+'f', self.params_names_list)[0]
            conv_b_name = _find_params('conv'+str(i+1)+'b', self.params_names_list)[0]
            print('\t\tCONV: setting '+ conv_W_name+' '+ conv_b_name)
            print('\t\tCONV: stride '+ str(_conv_stride[i]) + ', filter-group '+ str(_filtergroup_yn[i]))
            conv_W = self.params_values_list[self.params_names_list.index(conv_W_name)]
            conv_b = self.params_values_list[self.params_names_list.index(conv_b_name)]
            # batchnorm
            if _bnorm_yn[i]:
                bn_beta_name = _find_params('bn'+str(i+1)+'b', self.params_names_list)[0]
                bn_gamma_name = _find_params('bn'+str(i+1)+'m', self.params_names_list)[0]
                bn_moments_name = _find_params('bn'+str(i+1)+'x', self.params_names_list)[0]
                print('\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name)
                bn_beta = self.params_values_list[self.params_names_list.index(bn_beta_name)]
                bn_gamma = self.params_values_list[self.params_names_list.index(bn_gamma_name)]
                bn_moments = self.params_values_list[self.params_names_list.index(bn_moments_name)]
                bn_moving_mean = bn_moments[:,0]
                bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
            else:
                bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
            # set up conv "block" with bnorm and activation
            _set_convolutional(self.net, conv_W, np.squeeze(conv_b), _conv_stride[i],
                               bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,
                               filtergroup = _filtergroup_yn[i],
                               batchnorm = _bnorm_yn[i],
                               activation = _relu_yn[i]
                               )

            if _pool_stride[i]>0:
                self.net.add(nn.MaxPool2D(pool_size = _pool_sz, strides = _pool_stride[i]))
     
    def forward(self, z, x, params_names_list, params_values_list):
        net_z = self.net(z)
        net_x = self.net(x)
        print(net_z.shape)
        print(net_x.shape)
        out = self.match_templates(self, net_z, net_x, params_names_list, params_values_list)
        return out

    def match_templates(self, net_z, net_x):
        # out0, out1 shape: [B C H W]
        Bz, Cz, Hz, Wz = net_z.shape
        Bx, Cx, Hx, Wx = net_x.shape
#        assert Bz == Bx, ('Z and X should have same Batch size')
#        assert Cz == Cx, ('Z and X should have same Channels number')
            
        net_z = nd.reshape(data = net_z, shape = [Bz * Cz, 1, Hz, Wz])
        net_x = nd.reshape(data = net_x, shape = [1, Bz * Cz, Hx, Wx])
    
        net_final = nd.Convolution(data = net_x, 
                                   weight = net_z, 
                                   num_filter = Bz * Cz,
                                   kernel = [Hz, Wz],
                                   num_group = Bz * Cz,
                                   no_bias = True
                                   )
        
        net_final = nd.split(net_final, 3, axis=1)
        net_final = nd.concat(net_final[0], net_final[1], net_final[2], dim = 0)
        net_final = nd.expand_dims(nd.sum(net_final, axis = 1), axis = 1)
        net_final = self.bn_final(net_final)
            
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

def _set_convolutional(net, W, b, stride, bn_beta, bn_gamma, bn_mm, bn_mv, 
                      filtergroup = False, batchnorm = True, activation = True,
                      reuse = False):
    # use the input scope or default to "conv"
#    with net.name_scope():
        if filtergroup:
            net.add(nn.Conv2D(channels = W.shape[0],
                              in_channels = W.shape[1] * 2, 
                              kernel_size = W.shape[2],
                              strides = stride, 
                              groups = 2,
                              weight_initializer = SiamInit(W),
                              bias_initializer = SiamInit(b)
                             ))

        else:
            net.add(nn.Conv2D(channels = W.shape[0], 
                              in_channels = W.shape[1], 
                              kernel_size = W.shape[2],
                              strides = stride, 
                              weight_initializer = SiamInit(W),
                              bias_initializer = SiamInit(b)
                             ))
            
        if batchnorm:
            net.add(nn.BatchNorm(beta_initializer = SiamInit(bn_beta),
                                 gamma_initializer = SiamInit(bn_gamma),
                                 running_mean_initializer = SiamInit(bn_mm),
                                 running_variance_initializer = SiamInit(bn_mv),
                                 in_channels = W.shape[0],
                                 use_global_stats = True
                                 ))

        if activation:
            net.add(nn.Activation('relu'))
