# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:25:33 2016

@author: HP-PC
"""

import pandas as pd
import os
train=pd.read_excel(os.path.join(os.path.dirname("D:\MS11\machine learning")\
,'machine learning','blog-gender-dataset.xlsx'),sheetname="training",header=0\
,skip_footer=640,parse_cols=[0,1],na_values=[','],keep_default_na=False)
train.dropna(how='any')
#num_reviews = train["blog_text"].size
#print num_reviews
from nltk.corpus import stopwords
import re

def review_to_words( raw_review ):
    # Function to convert a raw text to a string of words
    # The input is a single string (a raw text), and 
    # the output is a single string (a preprocessed text)
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters
    
    
    letters_only = re.sub('[^a-zA-Z]'," ", raw_review)
    
    
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

#review_to_words( train["blog_text"][0])




# Get the number of reviews based on the dataframe column size
num_reviews = train["blog_text"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
   
    processed_review=review_to_words( train["blog_text"][i] )
    clean_train_reviews.append( processed_review )






print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 400) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag
    
    print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["gender"] )

# Read the test data
test=pd.read_excel(os.path.join(os.path.dirname("D:\MS11\machine learning")\
,'machine learning','test_data.xlsx'),sheetname="test",header=0\
,parse_cols=[0,1],na_values=[','],keep_default_na=False)

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["blog_text"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    
    clean_review = review_to_words( test["blog_text"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"gender":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "test1_data.csv", index=False, quoting=3 )
from sklearn.metrics import accuracy_score
y_true=pd.read_excel(os.path.join(os.path.dirname("D:\MS11\machine learning")\
,'machine learning','test_data.xlsx'),sheetname="test",header=0\
,parse_cols=[1],na_values=[','],keep_default_na=False)
y_true.as_matrix(columns=None);
y_pred=pd.read_csv("test1_data.csv",header=0,quoting=3)
value=accuracy_score(y_true, y_pred)
print value
correctly_classified=accuracy_score(y_true, y_pred, normalize=False)
print correctly_classified
