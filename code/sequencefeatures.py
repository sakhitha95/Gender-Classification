import pickle
import pandas as pd
import os
import nltk
import sys
import re
import collections
import numpy as np
import csv

with open("C:\Users\sakhi\Desktop\machine\otput2.txt", 'rb') as f:
    my_list = pickle.load(f)
print my_list
train=pd.read_excel(os.path.join(os.path.dirname("C:\Users\sakhi\Desktop\machine")
,'machine','dataset.xlsx'),sheetname="training",header=None
,parse_cols=[0,1],na_values=' ',keep_default_na=False)
t2=[]
for i in range(len(train)):
    text=nltk.word_tokenize(train[0][i])
    ta=nltk.pos_tag(text)
    t=""
    t1=[]
    for i in range(0,len(ta)):
        t1.append(ta[i][1])
    t=' '.join(t1)
    t2.append(t)
tf=[]
for seq in my_list:
    seql=seq.split(' ')
    coun=0
    t1=[]
#    t1.append(seq)
    for d in t2:
        coun=d.count(seq)
        t1.append(coun)
    tf.append(t1)
#print tf



a=np.asarray(tf)
np.savetxt("out.csv", a, delimiter=",")