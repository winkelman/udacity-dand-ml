#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tester import dump_classifier_and_data

with open("word_dataset.pkl", "r") as f_in:
        data_dict = pickle.load(f_in)

reduced_features = ['poi', 'exercised_stock_options', 'other', 'expenses',
							'holiday', 'total_stock_value', 'salary']


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(learning_rate = 0.55)

dump_classifier_and_data(ada, data_dict, reduced_features)