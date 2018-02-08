# -*-coding:utf-8 -*
import torch
import numpy as np
import preprocesser
import modules
from torch.autograd import Variable


#load dataset
def load_tsv(filename):
    '''load the rotten tomatoes sentiment dataset'''
    phID, stcID, ph, sentiment = [],[],[],[]
    with open(filename) as f: 
        l = f.readline()
        if (len(l.split("\t")) == 4):
            '''TRAIN dataset'''
            for line in f:
                l = line.split("\t")
                if (int(l[3][0]) < 2):
                    phID.append(l[0]), stcID.append(l[1]), ph.append(l[2]), sentiment.append(0)
                else:
                    phID.append(l[0]), stcID.append(l[1]), ph.append(l[2]), sentiment.append(1)
        else:
            for line in f:
                l = line.split("\t")
                phID.append(l[0]), stcID.append(l[1]), ph.append(l[2])
                
    return np.array(phID).astype(int), np.array(stcID).astype(int), np.array(ph), np.array(sentiment).astype(int)

#make validation dataset
def split_train_test(data, percentage=80):
    '''return indexes for a train/test split of the given data'''
    size = len(data)
    indexes = np.arange(0,size)
    np.random.shuffle(indexes)
    
    train_indexes = indexes[0:int(size*percentage/100)]
    test_indexes = indexes[int(size*percentage/100):-1]
    
    return train_indexes, test_indexes




#sample from Gumbel(0,1) = -log(-log(Uniform(0,1))
def gumbel():
    '''return a sample from the Gumbel Distribution: -log(-log(Uniform[0,1]))'''
    return -torch.log(-torch.log(torch.FloatTensor(2).uniform_()))

def r(logp, g, temperature):
    '''function to compute the r_i approximation (see training paragraph)'''
    num = [torch.exp( (pi + gi)/temperature) for (pi,gi) in zip(logp,g)]
    denum = torch.sum(torch.exp( (logp.data + g)/temperature))
    return [n.data/denum for n in num]
    

def X(p,g,i,temperature):
    '''function for exp(p+g/t) calculation (r derivative)'''
    return np.exp((p[i]+g[i])/temperature)
    
    
#fonctions d'évaluations
    
def accuracy(indexes,lang,targets,skim_rnn):
    '''compute the accuracy for a preprocessed lang corpus, its targets,
    and given a skim rnn model'''
    acc = 0.0
    for i in indexes:
        s,t =lang.corpus[i], targets[i]
        input,target =  preprocesser.makeInputTarget(lang, s, t)
        input_length = input.size()[0]
        hidden = skim_rnn.initHidden()
        for word in range(input_length):
            x = Variable(input[word])
            output, hidden, p, choice = skim_rnn(x, hidden)

        if output.exp().multinomial().data[0,0] == target.data[0]:
            acc += 1
            
    return 100*float(acc/len(indexes))
    
    
def accuracy_simple_rnn(indexes,lang,targets,rnn,embedder):
    '''compute a simpler accuracy for a classic RNN'''
    acc = 0.0
    for i in indexes:
        s,t =lang.corpus[i], targets[i]
        input,target =  preprocesser.makeInputTarget(lang, s, t)
        input_length = input.size()[0]
        hidden = rnn.initHidden()
        for word in range(input_length):
            x = Variable(input[word])
            embedding = embedder(x).view(1,1,-1)
            output, hidden = rnn(embedding, hidden)

        if output.exp().multinomial().data[0,0] == target.data[0]:
            acc += 1
            
    return 100*float(acc/len(indexes))
    



