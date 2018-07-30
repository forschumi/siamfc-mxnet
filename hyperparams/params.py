# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:58:03 2018

@author: pgao
"""

import json
from collections import namedtuple


def paramsInitial(in_params={}):

    with open('hyperparams/params.json') as json_file:
        params = json.load(json_file)              

    for name, value in in_params.items():
        params[name] = value
    
    params = namedtuple('params', params.keys())(**params)

    return params
