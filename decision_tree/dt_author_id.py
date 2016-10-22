#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from tools.email_preprocess import preprocess

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "Training time: {}".format(round(time() - t0, 3))

t1 = time()
pred = clf.predict(features_test)
print "Prediction time: {}".format(round(time() - t1, 3))

accuracy = accuracy_score(labels_test, pred)

print "Accuracy is {}".format(accuracy)

#########################################################
### your code goes here ###


#########################################################
