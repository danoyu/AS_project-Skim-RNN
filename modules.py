# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:56:41 2018

@author: pierre
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class Selector(nn.Module):
    '''The selector modules implements the Q decision maker of the paper. The logsoftmax
    output models the probability to skim or fully read the current word'''
    def __init__(self, input_size, n_choices=2):
        '''
            input_size : the size of the input
            n_choices : the number of RNNs in the following network. Default is 2.
        '''
        super(Selector, self).__init__()
        self.input_size = input_size
        self.n_choices = n_choices
        self.linear = nn.Linear(input_size, n_choices)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
        return self.softmax(self.linear(x))
    


class RNN(nn.Module):
    '''This module implements a standard RNN module with linear and logsoftmax 
    layers for classification purposes'''
    def __init__(self, input_size, hidden_size, output_size):
        '''
            input_size : the size of the input
            hidden_size : the size of the latent space
            output_size : number of classes in the output
            #todo : gru, lstm, cudas ? options
        '''
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        
        self.h20 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.h20(output)
        output = self.softmax(output.view(1,-1))
        return output, hidden
    
    def initHidden(self):
        return Variable(torch.rand(1, 1, self.hidden_size))
    
    
    