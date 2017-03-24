#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def get_words(enron_data):
	"""get words associated with positive labels.
	fits tfidf on topic text for all persons in data set.
	random forest with cross validation to get best vocab."""

	# only persons with topic data
	features = []
	labels = []
	for person in enron_data:
		text = enron_data[person].get('topics')
		if text:
			features.append(text)
			labels.append(enron_data[person]['poi'])

	tfidf = TfidfVectorizer(sublinear_tf=True,
							max_features=40, # vocab limit for all persons (40)
							max_df=0.20) # (.20)
	transformed_features = tfidf.fit_transform(features).toarray() # [num_obs x vocab]
	vocab = tfidf.get_feature_names()
	# need array for multiple indexing
	labels = np.array(labels)
	num_splits = 10 # iterations for cv
	#scores = np.zeros(len(vocab)) # empty array if cumulative scores
	f1 = 0 # initialize f1 score to zero
	sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.30)
	# not aggregating scores, just vocab from best score
	for train_idx, test_idx in sss.split(transformed_features, labels):
		train_feat, train_lab = transformed_features[train_idx], labels[train_idx]
		test_feat, test_lab = transformed_features[test_idx], labels[test_idx]
		rf = RandomForestClassifier(n_estimators=30)
		rf.fit(train_feat, train_lab)
		#scores += rf.feature_importances_ # add scores
		pred = rf.predict(test_feat)
		currrent_f1 = f1_score(test_lab, pred)
		if currrent_f1 > f1:
			f1 = currrent_f1
			scores = rf.feature_importances_
	#scores /= num_splits # average over splits if cumulative scores
	# get top 10 vocab for best fit
	feature_idx = scores.argsort()[::-1].tolist()[0:10]
	best_words = ([vocab[i] for i in feature_idx])
	return best_words


def make_words_feature(enron_data):
	"""create text features from best words.
	value is frequency of best word in person's topic corpus."""
	words_to_add = get_words(enron_data)
	for person in enron_data:
		for word in words_to_add:
			count = enron_data[person]['topics'].count(word)
			enron_data[person][word] = count
		# remove topics
		enron_data[person].pop('topics')
	return enron_data


def build_word_dataset(enron_data):
    word_feature_data = make_words_feature(enron_data)
    # export file
    with open("word_dataset.pkl", "w") as f_out:
        pickle.dump(word_feature_data, f_out)


if __name__ == '__main__':
	pass
	'''
	with open("topic_dataset.pkl", "r") as f_in:
		enron_data = pickle.load(f_in)
	print get_words(enron_data)
	#build_word_dataset(enron_data)
	#pprint.pprint(word_feature_data)
	'''