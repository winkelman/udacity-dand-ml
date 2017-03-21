#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../tools/")
from tester import test_classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# benchmark with decision tree and/or naive bayes?


def benchmark_score(enron_data, features_list):
	# email address in features before email corpus data set
	if 'email_address' in features_list:
		features_list.remove('email_address')
	# 'poi' is first in list for test_classifier
	if features_list[0] != 'poi':
		features_list.remove('poi')
		features_list = ['poi'] + features_list

	clf = DecisionTreeClassifier()
	test_classifier(clf, enron_data, features_list)
	'''
	clf = GaussianNB()
	test_classifier(clf, enron_data, features_list)
	'''