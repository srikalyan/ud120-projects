#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from sklearn.metrics import accuracy_score

from tools.email_preprocess import preprocess

from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(kernel="rbf", C=10000)


t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time() - t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time() - t1, 3), "s"

accuracy = accuracy_score(labels_test, pred)

print "Accuracy is {}".format(accuracy)


print len([ i for i in pred if i == 1])

