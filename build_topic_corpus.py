#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import time
import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# reference code
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html


def lda_decomp(emails_list):
	"""lda decomposition with tfidf on list of email strings for a single person.
	returns string of all topics."""
	# adjust params
	tfidf = TfidfVectorizer(sublinear_tf=True,
							max_features=100, # vocab/person (100)
							max_df=0.10) # (0.10)

	emails_transformed = tfidf.fit_transform(emails_list).toarray()
	# vocabulary
	vocab = tfidf.get_feature_names()
	# adjust params
	lda = LatentDirichletAllocation(n_topics=7, # topics/person (7)
								max_iter=20,
                                learning_method='online', # faster than batch
                                learning_offset=10, # downweight early iterations
                                random_state=27)

	lda.fit(emails_transformed)
	# max words for each topic
	max_topic_words = 10 # words/topic (10)
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


def build_topic_dataset(enron_data):
    # time to build corpus
    start_time = time.time()
    topic_data = make_topic_corpus(enron_data)
    elapsed_time = time.time() - start_time
    print "time to build topics: {} minutes".format(round(elapsed_time/60.0, 2))
    # export file
    with open("topic_dataset.pkl", "w") as f_out:
        pickle.dump(topic_data, f_out)


if __name__ == '__main__':
	pass
	'''
	with open("emails_dataset.pkl", "r") as f_in:
    	enron_data = pickle.load(f_in)
    build_topic_dataset(enron_data)
	#pprint.pprint(topic_data)
	'''