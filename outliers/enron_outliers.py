#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from tools.feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)

for k, v in data_dict.iteritems():
    if v['bonus'] != 'NaN' and v['bonus'] > 5000000 and v['salary'] > 1000000:
        print k, v

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)



### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# used the following to find total as outlier

# max = 0
# max_k = None
# max_v =None
#
# print data
#
# for k,v in data_dict.iteritems():
#     if v['salary'] != 'NaN' and max < v['salary']:
#         max = v['salary']
#         max_k = k
#         max_v = v
#
# print("The max salary {} with max k is {}, {}".format(max, max_k, max_v))
#
# for k,v in data_dict.iteritems():
#     if  v['bonus'] != 'NaN' and max < v['bonus']:
#         max = v['bonus']
#         max_k = k
#         max_v = v
#
# print("The max bonus {} with max k is {}, {}".format(max, max_k, max_v))
