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
    def __init__(self, input_size, hidden_size, output_size, cell='gru'):
        '''
            input_size : the size of the input
            hidden_size : the size of the latent space
            output_size : number of classes in the output
            cell : should be either gru, lstm, or linear
        '''
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell        
        self.h20 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        if self.cell=='linear':
            self.layer = nn.Linear(input_size,hidden_size)
        elif self.cell=='lstm':
            self.layer = nn.LSTM(input_size, hidden_size)
        else:
            self.layer = nn.GRU(input_size, hidden_size)
            
    def forward(self, input, hidden):
        
        if self.cell == 'lstm':
            _ , hidden = self.layer(input.view(1,1,-1), hidden)
            output = self.h20(hidden[0])
            output = self.softmax(output.view(1,-1))
            return output, hidden
            
        else:
            output, hidden = self.layer(input.view(1, 1, -1), hidden)
            output = self.h20(output)
            output = self.softmax(output.view(1,-1))
            return output, hidden
    
    def initHidden(self):
        if self.cell=='lstm':
            return (Variable(torch.rand(1, 1, self.hidden_size)), Variable(torch.rand(1, 1, self.hidden_size)))
        else:
            return Variable(torch.rand(1, 1, self.hidden_size))
        
        
        
class SkimRNN(nn.Module):
    
    def __init__(self, embedder, selector, rnn, skim_rnn):
        
        self.embedder = embedder
        self.selector = selector
        self.rnn = rnn
        self.skim_rnn= skim_rnn
        
        self.d = self.rnn.hidden_size
        self.dprime= self.skim_rnn.hidden_size
    
    def forward(self, input, hidden):
        
        embedding = self.embedder(input).view(1,1,-1)
        x = torch.cat((embedding, hidden),2).view(1,-1)
        p = self.selector(x)
        
        q = torch.multinomial(p.exp())
        choice = int(q.data[0,0])
        
        if choice == 0:
            #go through the normal RNN
            output, hidden = self.rnn(embedding, hidden)
        
        else:
            #go through the skim rnn, which implies 'cutting' the hidden state,
            #running through the neural network, and building the new hidden state.
            h0 = hidden.view(-1)[:self.dprime]
            output, h0 = self.skim_rnn(embedding, h0.view(1,1,-1))
            h1 = hidden.view(-1)[self.dprime:]
            hidden = torch.cat( (h0.view(1,1,-1 ), h1.view(1,1,-1)), 2) 
            
        return output, hidden
        
       
       
rnn = RNN(24,36,2,cell='lstm')
hidden = rnn.initHidden()
cell = rnn.initHidden()
x = Variable(torch.randn(24)).view(1,1,-1)

output, hidden = rnn(x, hidden)

        





    
    
    