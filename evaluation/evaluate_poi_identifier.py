#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys

from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,
                                                                                             labels,
                                                                                             test_size=0.3,
                                                                                             random_state=42)


### your code goes here
clf = DecisionTreeClassifier()
# parameters = {'min_samples_leaf':[10, 20, 30]}
# dt = DecisionTreeClassifier()
# clf = GridSearchCV(dt, parameters)

clf.fit(features_train, labels_train)

# print "The best parameters are ".format(sorted(clf.cv_results_.keys()))

predict_labels = clf.predict(features_test)

accuracy = accuracy_score(labels_test, predict_labels)

print "Accuracy is {}".format(accuracy)

print "The total number of pois predicted {}".format(len([ label for label in predict_labels if label]))
print "The total number of pois actual {}".format(len([ label for label in labels_test if label]))
print "The total number of people in test set {}".format(len(predict_labels))

true_positives = 0
for i in range(0, len(labels_test)):
    if predict_labels[i] and (predict_labels[i] == labels_test[i])  :
        true_positives+=1


print "The true positives are {}".format(true_positives)

print "The precision and recall scores are {} and {}".format(precision_score(labels_test, predict_labels),
                                                             recall_score(labels_test, predict_labels))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]


print "The precision and recall scores are {} and {}".format(precision_score(predictions, true_labels),
                                                             recall_score(predictions, true_labels))



