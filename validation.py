# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:03:00 2018

@author: pierre
"""

import pickle as pkl
import time
from torch.autograd import Variable
from torch import nn

import preprocesser
import outils


'''this file is a script for various tests, validation, and plots'''

phraseID, sentenceID, sentences, sentiment = outils.load_tsv("../train.tsv")
lang = preprocesser.Preprocesser({})
lang.load("preprocessing_RT")

#load indexes
test_indexes = pkl.load(open('models/test_indexes.p','rb'))

#load model
model = pkl.load(open('models/skimrnn_200_5.p','rb'))



#calculate accuracy 
start_time = time.time()
acc = outils.accuracy(test_indexes, lang, sentiment, model)
end_time = time.time()
print end_time - start_time
print acc


i = test_indexes[129]
s,t =uni_s, 0
print s
print t
input,target =  preprocesser.makeInputTarget(lang, s, t)
input_length = input.size()[0]
hidden = model.initHidden()
print input
for word in range(input_length):
    x = Variable(input[word])
    print input[word]
    output, hidden, p, choice = model(x, hidden)
    print p.exp().data[0,choice]
    print choice
    print 10*'#'
        
    
output.exp().multinomial().data[0,0]
            