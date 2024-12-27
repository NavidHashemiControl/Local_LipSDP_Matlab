# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 00:31:52 2024

@author: navid
"""

from new_compare_lp_naive_serial import compare_bounds
import numpy as np
import torch
import torch.nn as nn
from functions import  export2matlab

layer_configs = [(20, 20, 20, 1), ]
layer_config=layer_configs[0]

net=torch.load('Comparison.pt')





X = torch.Tensor(np.zeros((layer_config[0],1)))
epsilon=torch.Tensor([0.1])
epsilon=epsilon*np.ones(X.shape)
repeats = 1

compare_bounds(layer_configs,net,X,epsilon,repeats)