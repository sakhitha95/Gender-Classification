# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:32:43 2016

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
conversation = ['know', 'people', 'think', 'person', 'tell', 'feel', 'friends', 'talk',\
'new', 'talking', 'mean', 'ask', 'understand', 'feelings', 'care',\
'thinking', 'friend', 'relationship', 'realize', 'question', 'answer','saying']
home=['woke', 'home', 'sleep', 'today', 'eat', 'tired', 'wake', 'watch','watched', 'dinner',\
' ate', 'bed', 'day','house', 'tv', 'early', 'boring','yesterday', 'watching', 'sit']
family=['years', 'family', 'mother', 'children', 'father', 'kids', 'parents',\
'old', 'year', 'child', 'son', 'married', 'sister', 'dad', 'brother',\
'moved', 'age', 'young', 'months', 'three', 'wife', 'living', 'college',\
'four', 'high', 'five', 'died', 'six', 'baby', 'boy', 'spend',\
'Christmas']
food=['food', 'eating', 'weight', 'lunch', 'water', 'hair', 'life', 'white',\
'wearing', 'color', 'ice', 'red', 'fat', 'body', 'black', 'clothes',\
'hot', 'drink', 'wear', 'blue', 'minutes', 'shirt', 'green', 'coffee',\
'total', 'store', 'shopping']
romance=['forget', 'forever', 'remember', 'gone', 'true', 'face', 'spent',
'times', 'love', 'cry', 'hurt', 'wish', 'loved']
positive=['absolutely', 'abundance', 'ace', 'active', 'admirable', \
'adore',\
'agree', 'amazing', 'appealing', 'attraction', 'bargain', 'beaming',\
'beautiful', 'best', 'better', 'boost', 'breakthrough', 'breeze',\
'brilliant', 'brimming', 'charming', 'clean', 'clear', 'colorful',\
'compliment', 'confidence', 'cool', 'courteous', 'cuddly', 'dazzling',\
'delicious', 'delightful', 'dynamic', 'easy', 'ecstatic',\
'efficient', 'enhance', 'enjoy','enormous', 'excellent', 'exotic',\
'expert', 'exquisite', 'flair','free', 'generous', 'genius', 'great',\
'graceful', 'heavenly', 'ideal', 'immaculate', 'impressive', 'incredible',\
'inspire', 'luxurious', 'outstanding', 'royal', 'speed',\
'splendid', 'spectacular', 'superb', 'sweet', 'sure', 'supreme',\
'terrific', 'treat', 'treasure', 'ultra', 'unbeatable', 'ultimate',\
'unique', 'wow', 'zest']
negative=['wrong', 'stupid', 'bad', 'evil', 'dumb', 'foolish', 'grotesque',\
'harm', 'fear', 'horrible', 'idiot', 'lame', 'mean', 'poor', 'heinous',\
'hideous', 'deficient', 'petty', 'awful', 'hopeless', 'fool', 'risk',\
'immoral', 'risky', 'spoil', 'spoiled', 'malign', 'vicious', 'wicked',\
'fright', 'ugly', 'atrocious', 'moron', 'hate', 'spiteful', 'meager',\
'malicious', 'lacking']
emotion=['aggressive', 'alienated', 'angry', 'annoyed', 'anxious', 'careful',\
'cautious', 'confused', 'curious', 'depressed', 'determined',\
'disappointed','discouraged', 'disgusted', 'ecstatic', 'embarrassed',\
'enthusiastic', 'envious', 'excited', 'exhausted',\
'frightened', 'frustrated', 'guilty', 'happy', 'helpless', 'hopeful',\
'hostile', 'humiliated', 'hurt', 'hysterical', 'innocent', 'interested',\
'jealous', 'lonely', 'mischievous', 'miserable', 'optimistic',\
'paranoid', 'peaceful', 'proud', 'puzzled', 'regretful','relieved',\
'sad', 'satisfied', 'shocked', 'shy', 'sorry', 'surprised', 'suspicious',\
'thoughtful', 'undecided', 'withdrawn']
tarr=[]

features=['conversation','home','family','food','romance','positive','negative','emotion']
##tra=np.asarray(tarr)
for i in range(len(train)):
    text=nltk.word_tokenize(train[0][i])
    cnt=[]
    tarr1=[]
    for f in features:
        
        for word in f:
            cnt.append(text.count(word))
    
        tarr1.append(sum(cnt))
    tarr.append(tarr1)
tarr=np.asarray(tarr)
print tarr


a=np.asarray(tarr)
np.savetxt("basefeatures.csv", a, delimiter=",")