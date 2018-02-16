
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:24:46 2016

@author: sakhi
"""



# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
import nltk
import sys
import re
import collections
import numpy as np
import csv

##from nltk.corpus import brown
##from nltk import sent_tokenize, word_tokenize, pos_tag
##from nltk import pos_tag_sents
##brown_corpus = brown.raw()
##brown_sents = sent_tokenize(brown_corpus)
##brown_words = [word_tokenize(i) for i in brown_sents]
##brown_words = [word_tokenize(s) for s in sent_tokenize(brown.raw())]
##brown_tagged = [pos_tag(word_tokenize(s)) for s in sent_tokenize(brown.raw())]
##brown_tagged = pos_tag_sents([word_tokenize(s) for s in sent_tokenize(brown.raw())])
train=pd.read_excel(os.path.join(os.path.dirname("C:\Users\sakhi\Desktop\machine")
,'machine','blog-gender-dataset.xlsx'),sheetname="training",header=None
,skip_footer=640,parse_cols=[0,1],na_values=' ',keep_default_na=False)
##test=pd.read_excel(os.path.join(os.path.dirname("C:\Users\HONEY\Desktop")
##,'Desktop','test_data.xlsx'),sheetname="training",header=None
##,parse_cols=[0,1],na_values=' ',keep_default_na=False)
##print train[0][0]
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
##print text
print "completed reading"
taglist=set(["''","(",")",",","--",".",":","CC","CD","DT","EX","FW","IN","JJ",
             "JJR","JJS","LS","MD","NN","NN","NNP","NNPS","PDT","POS","PRP",
             "PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN"
             ,"VBP","VBZ","WDT","WP","WP$","WRB"])
seqp=set()

for e in taglist:

    seqp.add(e)
print seqp
print len(t2)
#count=0

def adherence(ad):
    count=0.0
    count1=0.0
    count2=0.0
    sunn=0.0
    tadx=nltk.word_tokenize(ad)
    for d in t2:
        m=re.search(re.escape(ad),d)
        if m:
            count=count+1
    for l in range(0,len(tadx)-1):
        for d in t2:
            tad1=tadx[0:l+1]
            tad1x=' '.join(tad1);
 #           print tad1
            m1=re.search(re.escape(tad1x),d)
            if m1:
                count1=count1+1
            
            tad2=tadx[l+1:len(tadx)]
            tad2x=' '.join(tad2)
            m2=re.search(re.escape(tad2x),d)
            if m2:
                count2=count2+1
        sunn=((count1*count2)/((len(t2))*(len(t2))))+sunn
##        print sunn
##        print count1
##        print count2
        count1=0.0
        count2=0.0
    if sunn>0:
##        print 'count'
##        print (count*count)/(len(t2)*len(t2))
##        print 'denom'
##        print sunn/(len(tadx)-1)
#        print ((count*count)/(len(t2)*len(t2)))/(sunn/(len(tadx)-1))
        return float(((count*count)/(len(t2)*len(t2)))/(sunn/(len(tadx)-1)))
    else:
        return 0
            


def posseq(pseq):
    global seqp
#    print pseq
#    global count
    for e in pseq:
        count=0.0
        for d in t2:
            m=re.search(re.escape(e),d)
            if m:
                count=count+1
##                print m.group(0)
#        print count
        if float(count/len(t2))>.3:
            if adherence(e)>.2:
                seqp.add(e)
   
    return
#posseq(seqp)
print seqp



for k in range(2,3):
  
    sp=set()
    spp=set()
    for e in seqp:
       
        spp.add(e)
    
    for e in spp:
        for d in taglist:
            l=d+' '+e
            sp.add(l)
        posseq(sp)
#    print seqp
    print 'hello'

print seqp   
seqpf=list(seqp)

# Open File

#
#f=open("C:\Users\sakhi\Desktop\machine\otput.csv", "wb")
#wr = csv.writer(f,dialect='excel')
#for item in seqpf: 
#    wr.writerow(item)
#print 'gfhg'
import pickle
with open("C:\Users\sakhi\Desktop\machine\otput.txt", 'wb') as f:
    pickle.dump(seqpf, f)
    
    
for k in range(3,4):
  
    sp=set()
    spp=set()
    for e in seqp:
       
        spp.add(e)
    
    for e in spp:
        for d in taglist:
            l=d+' '+e
            sp.add(l)
        posseq(sp)
#    print seqp
    print 'hello'

print seqp   
seqpf=list(seqp)

with open("C:\Users\sakhi\Desktop\machine\otput1.txt", 'wb') as f:
    pickle.dump(seqpf, f)

for k in range(4,5):
  
    sp=set()
    spp=set()
    for e in seqp:
       
        spp.add(e)
    
    for e in spp:
        for d in taglist:
            l=d+' '+e
            sp.add(l)
        posseq(sp)
#    print seqp
    print 'hello'

print seqp   
seqpf=list(seqp)


with open("C:\Users\sakhi\Desktop\machine\otput2.txt", 'wb') as f:
    pickle.dump(seqpf, f)
    
for k in range(5,6):
  
    sp=set()
    spp=set()
    for e in seqp:
       
        spp.add(e)
    
    for e in spp:
        for d in taglist:
            l=d+' '+e
            sp.add(l)
        posseq(sp)
#    print seqp
    print 'hello'

print seqp   
seqpf=list(seqp)


with open("C:\Users\sakhi\Desktop\machine\otput3.txt", 'wb') as f:
    pickle.dump(seqpf, f)
    
for k in range(6,7):
  
    sp=set()
    spp=set()
    for e in seqp:
       
        spp.add(e)
    
    for e in spp:
        for d in taglist:
            l=d+' '+e
            sp.add(l)
        posseq(sp)
#    print seqp
    print 'hello'

print seqp   
seqpf=list(seqp)


with open("C:\Users\sakhi\Desktop\machine\otput4.txt", 'wb') as f:
    pickle.dump(seqpf, f)