#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_feature_scores(enron_data):
    """get highest scoring features from decision tree classifier
    using stratified cross validation."""

    # pandas to get vectorized labels, features
    df = pd.DataFrame.from_dict(enron_data, orient = 'index')
    # classifiers can't handle np.nan
    df = df.replace('NaN', 0)
    labels = df.poi.values
    features = df[df.columns.difference(['poi', 'email_address'])] # NEGATION POI ONLY HERE
    var_names = features.columns.tolist()
    features = features.values

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.30)
    # RF or DT?
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, {'n_estimators': [1000]}, scoring='f1', cv=sss)
    #dt = DecisionTreeClassifier()
    #clf = GridSearchCV(dt, {'max_depth': [10]}, scoring='f1', cv=sss)
    clf.fit(features, labels)
    #print clf.best_score_
    clf = clf.best_estimator_
    feature_idx = clf.feature_importances_.argsort()[::-1].tolist()
    best_features = ([var_names[i] for i in feature_idx])
    #print 'best features descending: {}'.format(best_features)
    return [var_names, clf.feature_importances_, best_features]


if __name__ == "__main__":
    pass
    with open("word_dataset.pkl", "r") as f_in:
        data_dict = pickle.load(f_in)
    get_features(data_dict)