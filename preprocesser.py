import numpy as np
import unicodedata
import string
import re
import random
import pickle as pkl


class Preprocesser():
    
    '''main functions'''
    def __init__(self,corpus):
        '''corpus : np_array(string)'''
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.corpus = corpus
        self.size = len(corpus)
    
    # Lowercase, trim, and remove non-letter characters
    # (no stop words in skim rnn ?)
    def normalize(self):
        new_corpus = np.array([])
        steps,i = np.arange(0,self.size,self.size/10),0
        for s in self.corpus:
            uni_s = s.tostring().decode('unicode-escape')
            uni_s = self.unicodeToAscii(uni_s.lower().strip())
            uni_s = re.sub(r"([.!?])", r" \1", uni_s)
            uni_s = re.sub(r"[^a-zA-Z.!?]+", r" ", uni_s)
            new_corpus = np.append(new_corpus, uni_s)
            if (i in steps):
                print ("...")
            i+=1
        self.corpus = new_corpus


    def addSentences(self):
        for sentence in self.corpus:
            self.addSentence(sentence)
        
    '''called within the module:'''
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
    def save(self,filename):
        pkl.dump(self.corpus,open(filename+"_corpus.pkl",'wb'))
        pkl.dump(self.word2count,open(filename+"_w2c.pkl",'wb'))
        pkl.dump(self.word2index,open(filename+"_w2i.pkl",'wb'))
        pkl.dump(self.index2word,open(filename+"_i2w.pkl", 'wb'))
    
    def load(self, filename):
        self.corpus = pkl.load(open(filename+"_corpus.pkl", 'rb'))
        self.word2count = pkl.load(open(filename+"_w2c.pkl", 'rb'))
        self.word2index = pkl.load(open(filename+"_w2i.pkl", 'rb'))
        self.index2word = pkl.load(open(filename+"_i2w.pkl", 'rb'))
        self.size = len(self.corpus)
        self.n_words = len(self.word2count.keys())
        

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1, 1)
    return result


def makeInputTarget(lang, sentence, target, n_classes=2):
    input_variable = variableFromSentence(lang, sentence)
    if target >= n_classes:
        print 'target not in range (0, #classes - 1)'
        return -1
    target_variable = Variable(torch.LongTensor([target]))
    return (input_variable, target_variable)



 
