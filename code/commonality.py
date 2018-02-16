# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:28:10 2016

@author: sakhi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:20:46 2016

@author: sakhi
"""
import pickle
import pandas as pd
import os
import nltk
from nltk.corpus import wordnet
import sys
import re
import collections
import numpy as np
import csv
from itertools import chain
train=pd.read_excel(os.path.join(os.path.dirname("C:\Users\sakhi\Desktop\machine")
,'machine','dataset.xlsx'),sheetname="training",header=None
,parse_cols=[0,1],na_values=' ',keep_default_na=False)
t2=[]


tarr=[]

commonalityp=['Centrality','Cooperation','Rapport']
commonalityn=['Diversity','Exclusion' ,'Liberation']



for i in range(len(train)):
    text=nltk.word_tokenize(train[0][i])
    tarr1=[]
    for  fet in commonalityp:
        cnt=[]
        synonyms = wordnet.synsets(fet)
        synonym= set(chain.from_iterable([word.lemma_names() for word in synonyms]))

        for word in synonym:
            cnt.append(text.count(word))
        tarr1.append(sum(cnt))
 
    tarr.append(sum(tarr1))
a=np.asarray(tarr)
print a
tar1=[]
for i in range(len(train)):
    text=nltk.word_tokenize(train[0][i])
    tarr1=[]
    for  fet in commonalityn:
        cnt=[]
        synonyms = wordnet.synsets(fet)
        synonym= set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        for word in synonym:
            cnt.append(text.count(word))
        tarr1.append(sum(cnt))
 
    tar1.append(sum(tarr1))
b=np.asarray(tar1)

p=np.asarray(a-b)
np.savetxt("commonalityfeatures.csv", a, delimiter=",")