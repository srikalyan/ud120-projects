#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

# 33614 this number is the index got for feature importance with threshold > 0.2 with older version of text learning output
# print "the world that is causing underfitting is {}".format(vectorizer.get_feature_names()[33614]) # this returns sshacklensf

# 14343 this number is the index got for feature importance with threshold > 0.2 with newer version of text learning output
# print "the world that is causing underfitting is {}".format(vectorizer.get_feature_names()[14343]) # this returns cgermannsf

# 21323 this number using the same above steps for 3rd time
print "the world that is causing underfitting is {}".format(vectorizer.get_feature_names()[21323]) # this returns houectect

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### your code goes here
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()

t0 = time()
clf.fit(features_train, labels_train)
print "Training time: {}".format(round(time() - t0, 3))

feature_importances = clf.feature_importances_
for i in range(0, len(feature_importances)):
    if feature_importances[i] > 0.2:
        print "features with importance > 0.2 is at index {} with value {}".format(i, feature_importances[i])

t1 = time()
pred = clf.predict(features_test)
print "Prediction time: {}".format(round(time() - t1, 3))

accuracy = accuracy_score(labels_test, pred)

print "Accuracy is {}".format(accuracy)

