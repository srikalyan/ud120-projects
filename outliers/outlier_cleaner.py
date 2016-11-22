#!/usr/bin/python

import heapq

CLEANING_PERCENT = 10


def nth_largest(n, iter):
    return heapq.nlargest(n, iter)[-1]


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    error_data = []
    for i in range(0, len(predictions)):
        error_data.append(abs(predictions[i] - net_worths[i]))

    k = (len(ages) * 10) / 100
    largest_k = nth_largest(k, error_data)

    for i in range(0, len(predictions)):
        if error_data[i] < largest_k:
            cleaned_data.append((ages[i], net_worths[i], error_data[i]))
    ### your code goes here
    return cleaned_data
