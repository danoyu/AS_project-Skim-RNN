# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:25:43 2018

@author: pierre
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import time

import preprocesser
import outils
import modules


'''this file is the main file of the project, it trains a skim rnn given inputs and targets'''

'''load and split in datasets the data'''
phraseID, sentenceID, sentences, sentiment = outils.load_tsv("../train.tsv")
train_indexes, test_indexes = outils.split_train_test(phraseID)


'''load (or make) the preprocesser)'''
#lang = preprocesser.Preprocesser(sentences)
#lang.normalize()
#lang.addSentences()
#lang.save("preprocessing_IMDB")
lang = preprocesser.Preprocesser({})
lang.load("preprocessing_RT")


'''train the model'''


#size parameters : the embedding dimension, the main rnn and small rnn sizes
embedding_size = 512
d = 200
dprime = 5

#modules : an embedder, a selector, and two rnns that will make a skim-RNN (see modules.py for details)
embedder = nn.Embedding(lang.n_words+2,embedding_size)
selector = modules.Selector(embedding_size+d)
small_rnn = modules.RNN(embedding_size,dprime,2)
rnn = modules.RNN(embedding_size,d,2)
skim_rnn = modules.SkimRNN(embedder, selector, rnn, small_rnn)

#loss criteria and optimizers : selector optimization done by hand, only the rnn parameters must be learned through torch.optim
#                               loss criterion is NLLLoss for the classification problem
criterion = nn.NLLLoss()
optim_rnn = optim.Adam([{'params':skim_rnn.rnn.parameters()},{'params':skim_rnn.small_rnn.parameters()}], lr=1e-2)
selector_lr = 1e-2

#other variables : hyperparameters temperature and gamma, arrays for storing purposes, maximum of epochs, ...
temperature, gamma = 0.25,1e-2
global_losses, accs = [], []
epoch,mod, max_epoch=0,10,50
start_time = time.time()

#training snippet
for epoch in range(max_epoch):
    #go through indexes
    for i in train_indexes[epoch*500:500+epoch*500]:
        #reset a bunch of variables
        logp_skim = []
        loss, selector_loss= 0,0
        hidden = skim_rnn.initHidden()
        optim_rnn.zero_grad()
        #make the input and the target
        sentence, target = lang.corpus[i], sentiment[i]
        input, target = preprocesser.makeInputTarget(lang,sentence,target)
        input_size = input.size()[0]

        #go through each word
        for word in range(input_size):
            x = Variable(input[word])
            output, hidden, p, choice = skim_rnn(x, hidden)
            logp_skim.append(p.data[0,1])     #skimming proba, for regularization purposes
            
            #compute losses
            loss += criterion(output,target)
            g = outils.gumbel()
            if choice==0:
                selector_loss += loss.data*( - outils.X(p.view(-1).data,g,0,temperature) / ( (temperature*np.exp(p.view(-1).data[1]))*(outils.X(p.view(-1).data,g,0,temperature) + outils.X(p.view(-1).data,g,1,temperature))) )
            else:
                selector_loss += loss.data*( - outils.X(p.view(-1).data,g,1,temperature) / ( (temperature*np.exp(p.view(-1).data[0]))*(outils.X(p.view(-1).data,g,0,temperature) + outils.X(p.view(-1).data,g,1,temperature))) )
        
        #final losses calculations, backwards, and gradient steps
        selector_loss = Variable(selector_loss) + (gamma/input_size) * np.sum(logp_skim)
        loss.backward()

        skim_rnn.selector.linear.weight.data -= selector_loss.data*selector_lr
        optim_rnn.step()
        

    #store some losses and accuracies calculations
    if epoch%mod == 0:
        global_losses.append((loss.data[0], selector_loss.data[0]))
        accs.append(outils.accuracy(test_indexes[:500], lang, sentiment, skim_rnn))
        print epoch
        
    epoch += 1  
    
end_time = time.time()

print end_time - start_time


'''
print round(100*float(skim_rnn.skimcount)/skim_rnn.count,2)



#gamma_skim_rate_plot
plt.figure(0)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.linspace(1e-2,8e-1,10), global_skims, 'b-')
ax2.plot(np.linspace(1e-2,8e-1,10), global_accs, 'g-')

ax1.set_xlabel('gamma (coefficient de regularisation pour encourager le skimrate)')
ax1.set_ylabel('skimrate (%)', color='b')
ax2.set_ylabel('accuracy (%)', color='g')

ax1.set_ylim(0.3,0.65)
ax2.set_ylim(50,80)
plt.show()

'''


