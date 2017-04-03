#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def get_feature_scores(enron_data):
    """get highest scoring features from decision tree classifier
    using stratified cross validation."""

    # pandas to get vectorized labels, features
    df = pd.DataFrame.from_dict(enron_data, orient = 'index')
    # classifiers can't handle np.nan
    df = df.replace('NaN', 0)
    labels = df.poi.values
    features = df[df.columns.difference(['poi', 'email_address'])]
    # get names first
    var_names = features.columns.tolist()
    features = features.values

    num_splits = 10
    f1 = 0
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.30)
    # no aggregate scores, score is from best cross validated predictor
    for train_idx, test_idx in sss.split(features, labels):
        train_feat, train_lab = features[train_idx], labels[train_idx]
        test_feat, test_lab = features[test_idx], labels[test_idx]
        rf = RandomForestClassifier(n_estimators=30)
        rf.fit(train_feat, train_lab)
        pred = rf.predict(test_feat)
        currrent_f1 = f1_score(test_lab, pred)
        if currrent_f1 > f1:
            f1 = currrent_f1
            scores = rf.feature_importances_
    
    # get top 10 vocab
    scores_idx = scores.argsort()[::-1].tolist()
    sorted_names = ([var_names[i] for i in scores_idx])
    return [sorted_names, sorted(scores, reverse=True)]


if __name__ == "__main__":
    pass
    '''
    with open("word_dataset.pkl", "r") as f_in:
        data_dict = pickle.load(f_in)
    print get_feature_scores(data_dict)
    '''