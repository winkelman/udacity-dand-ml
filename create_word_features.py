#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def get_words(enron_data):
	"""get words associated with positive labels.
	fits tfidf on topic text for all persons in data set.
	grid search using random forest and cross validation
	to get best estimator and best vocab."""

	'''
	# try pandas/numpy arrays
	df = pd.DataFrame.from_dict(data_dict, orient = 'index')
    df = df.replace('NaN', np.nan)
    # OR?
    #df = df.replace('NaN', 0)
    labels = df.poi.values
    features = df.loc[:, df.columns != 'poi'].values
    '''

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
	transformed_features = tfidf.fit_transform(features).toarray()
	vocab = tfidf.get_feature_names()
	# stratified split, equal class ratio
	sss = StratifiedShuffleSplit(n_splits=10, test_size=0.30)
	# optimize for precision and recall
	# RF or DT?
	rf = RandomForestClassifier()
	clf = GridSearchCV(rf, {'n_estimators': [10000]}, scoring='f1', cv=sss)
	#dt = DecisionTreeClassifier()
	#clf = GridSearchCV(dt, {'max_depth': [10]}, scoring='f1', cv=sss)
	clf.fit(transformed_features, labels)
	#print clf.best_score_
	# best classifier
	clf = clf.best_estimator_
	# get top 10 vocab
	feature_idx = clf.feature_importances_.argsort()[::-1].tolist()[0:10]
	best_words = ([vocab[i] for i in feature_idx])
	#print best_words
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
	build_word_dataset(enron_data)
	#pprint.pprint(word_feature_data)
	'''