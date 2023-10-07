# %% 
from io import open 
import unicodedata
import re
import random
import numpy as np 

# %%
SOS_token = 0
EOS_token = 1


class Lang():
    '''
    Lang class is used to convert word to index and 
    index to word and as well as the count of each word 
    SOS = Star Of Sentence
    EOS = End Of Sentence 
    '''
    def __init__(self,name):
        self.name = name 
        self.word2index = {}
        self.index2word = {0: 'SOS',1:'EOS'}
        self.word2count = {}
        self.n_words = 2
    
    def add_Sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    
    def add_word(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] =1
            self.index2word[self.n_words] = word 
            self.n_words +=1
        else:
            self.word2count[word] +=1
# %%             
def normazilze_string_en(word):
    word = word.lower()
    word = re.sub(r"[^a-zA-Z!?]+", r" ", word)
    
    return word

def normazilze_ta(word):
    word = word.lower()
    return  re.sub(r'[?|$|.|!]',r'',word)


#%%
def readLang(lang1,lang2):
    print("Reading lines...")
    
    lang1_lines = open("E:\SeqtoSeq\data\eng_Latn-tam_Taml\\train.eng_Latn",encoding='utf-8')
    lang2_lines = open('E:\SeqtoSeq\data\eng_Latn-tam_Taml\\train.tam_Taml',encoding='utf-8')

    pairs_lang1 = [[normazilze_string_en(s) for s in l.split('\t')]for l in lang1_lines ]
    pairs_lang2 = [[normazilze_ta(s) for s in l.split('\t')]for l in lang2_lines]
    
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    
    return input_lang,output_lang,pairs_lang1,pairs_lang2 

def prepareData(lang1,lang2):
    input_lang,output_lang,pairs_eng,pairs_ta = readLang(lang1,lang2)
    
   
    [input_lang.add_Sentence(pair[0]) for pair in pairs_eng]
    [output_lang.add_Sentence(pair[0]) for pair in pairs_ta]
    
    return  input_lang,output_lang
        
    

    








    
