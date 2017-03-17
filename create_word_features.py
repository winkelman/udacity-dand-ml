#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_words(enron_data):
	"""get words associated with positive labels.
	fits tfidf on topic text for all persons in data set.
	grid search using random forest and cross validation
	to get best estimator and best vocab."""
	features = []
	labels = []
	for person in enron_data:
		text = enron_data[person].get('topics')
		if text:
			features.append(text)
			labels.append(enron_data[person]['poi'])

	vectorizer = TfidfVectorizer(sublinear_tf=True,
								max_features=200, # vocab limit for all persons
								max_df=0.10)
	transformed_features = vectorizer.fit_transform(features).toarray()
	vocab = vectorizer.get_feature_names()
	# stratified split, equal class ratio
	sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
	rf = RandomForestClassifier()
	# optimize for precision and recall
	clf = GridSearchCV(rf, {'n_estimators': [1000]}, scoring='accuracy', cv=sss)
	clf.fit(transformed_features, labels)
	# best classifier
	clf = clf.best_estimator_
	# get top 5 vocab
	feature_idx = clf.feature_importances_.argsort()[::-1].tolist()[0:5]
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


if __name__ == '__main__':
	#pass
	enron_data = pickle.load(open("topic_dataset.pkl", "r"))
	word_feature_data = make_words_feature(enron_data)
	pickle.dump(word_feature_data, open("word_feature_dataset.pkl", "w"))
	#pprint.pprint(word_feature_data)