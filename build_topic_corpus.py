#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import time
import pprint

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# reference code
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html


def lda_decomp(emails_list):
	"""lda decomposition with tfidf on list of email strings for a single person.
	returns string of all topics."""
	# adjust params
	tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,
								max_features=200, # vocab limit per person
								max_df=0.10) # LOW

	emails_transformed = tfidf_vectorizer.fit_transform(emails_list).toarray()
	# vocabulary
	vocab = tfidf_vectorizer.get_feature_names()
	# adjust params
	lda = LatentDirichletAllocation(n_topics=10, # ADJUST
								max_iter=20, # num iterations
                                learning_method='online', # faster than batch
                                learning_offset=10, # downweight early iterations
                                random_state=27)

	lda.fit(emails_transformed)
	# max words for each topic
	max_topic_words = 20
	# list of vocab topic strings
	topic_list = []
	# lda components are numpy 2D array
	for topic_idx, topic in enumerate(lda.components_):
		# sorted indices(ascending), reversed, to list, max words only
		top_vocab_idx = topic.argsort()[::-1].tolist()[:max_topic_words]
		# to string
		topic = ' '.join([vocab[i] for i in top_vocab_idx])
		topic_list.append(topic)
	# return all topics as string
	return ' '.join(topic_list)


def make_topic_corpus(enron_data):
	"""create a topic corpus for each person by lda decomposition"""
	for person in enron_data:
		corpus = enron_data[person].get('corpus')
		# has email corpus
		if corpus:
			topics = lda_decomp(corpus)
			enron_data[person]['topics'] = topics
		else:
			enron_data[person]['topics'] = ''
		# remove corpus
		enron_data[person].pop('corpus')
	return enron_data


if __name__ == '__main__':
	pass
	# time to build corpus
	'''
	start_time = time.time()
	enron_data = pickle.load(open("corpus_dataset.pkl", "r"))
	topic_data = make_topic_corpus(enron_data)
	elapsed_time = time.time() - start_time
	print "time to build topics: {} minutes".format(round(elapsed_time/60.0, 2))
	pickle.dump(topic_data, open("topic_dataset.pkl", "w"))
	#pprint.pprint(topic_data)
	'''