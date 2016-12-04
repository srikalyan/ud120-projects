#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,
                                                                                             labels,
                                                                                             test_size=0.3,
                                                                                             random_state=42)


### it's all yours from here forward!  

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

predict_labels = clf.predict(features_test)

accuracy = accuracy_score(labels_test, predict_labels)

print "Accuracy is {}".format(accuracy)
