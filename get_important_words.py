#!/usr/bin/python
# -*- coding: utf-8 -*-


## HERE WE TRY TO PREDICT IMPORTANT WORDS IN IDENTIFYING POI'S.
## WE SPLIT 'WORD' FEATURE AND 'POI' LABEL DATA INTO A 70-30 TRAIN/TEST SPLIT TO PREVENT OVERFITTING.


from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.tree import DecisionTreeClassifier


## PASS IN DATA_DICT AND RETURN A LIST OF IMPORTANT WORDS

def get_words(data_dict):
	
	## getting features and labels lists
	features = []
	labels = []
	
	for person in data_dict:
		poi = data_dict[person]['poi']
		words = data_dict[person]['words']
		labels.append(poi)
		features.append(words)


	## test/train split
        ## note that we are using the same random state number here so that we use the same training data for optimization of parameters used in gridsearchCV (poi_id.py)
        ## and feature selection with k-best (feature_selector.py)
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


	## idf text vectorization
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.05, stop_words='english')
	features_train_transformed = vectorizer.fit_transform(features_train)
	## transform test features based on training features
	features_test_transformed = vectorizer.transform(features_test)
	
	
	## convert to array for DT	
	features_train_transformed = features_train_transformed.toarray()
	features_test_transformed  = features_test_transformed.toarray()


	## fitting decision tree classifier
	clf = DecisionTreeClassifier()
	clf = clf.fit(features_train_transformed, labels_train)
	acc = clf.score(features_test_transformed, labels_test)
	print "\nAccuracy of word-based DT classifier on test data is: ", acc


	## extracting important features/words from DT classifier
	ft_imp = clf.feature_importances_.tolist() ## a vector of numerical importance for each feature/word
	most_imp = [] ## a list of pairs of numerical importance value and respective index (within 'ft_imp')
	for f in ft_imp:
		if f > 0.01: ## anything that is at all important has a numerical importance value greater than 0
			most_imp.append((f,ft_imp.index(f)))

	print "Number of important/suspicious words: ", len(most_imp)

	ftrs_list = vectorizer.get_feature_names()  ## a list of all features/words
	#print len(ftrs_list), len(ft_imp), most_imp ## double check our array lengths and indices
	most_imp.sort(key=lambda x: x[0], reverse=True) ## sorting by importance value from high to low
	best_features = []
	for tuple in most_imp: ## using index from 'most_imp' (contains index from 'ft_imp') to get actual word in 'ftrs_list'
	    best_features.append(ftrs_list[tuple[1]])


	return best_features
	
	