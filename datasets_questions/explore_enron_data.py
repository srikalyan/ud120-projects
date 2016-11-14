#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from final_project.poi_email_addresses import poiEmails

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_emails = poiEmails()

# we have 146 data points and each data point as 21 features

# total number of POIs not considering emails
poi_unfiltered = [ (poi, v) for poi, v in enron_data.items() if v["poi"] == 1 ]
print "pois from dataset {}".format(len(poi_unfiltered))

print len(poi_unfiltered)

poi_names = open("../final_project/poi_names.txt", "r").readlines()
poi_y = [name for name in poi_names if name.startswith("(y)") or name.startswith("(n)")]
print "poi names count {}".format(len(poi_y))

# poi_using_email = [ (poi, v) for poi, v in enron_data.items() if v["poi"] == 1 and v["email_address"] in poi_emails ]
#
# print len(poi_using_email)

# total value of stock belonging to james prentice
# print enron_data["PRENTICE JAMES"]["total_stock_value"]

# How many email messages do we have from Wesley Colwell to persons of interest?
# print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# What's the value of stock options exercised by Jeffrey K Skilling ?
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

