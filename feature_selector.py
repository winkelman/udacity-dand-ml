#!/usr/bin/python
# -*- coding: utf-8 -*-


from sklearn.linear_model import RidgeClassifierCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


## UNIVARIATE STATISTICAL ANALYSIS, RECURSIVE FEATURE ELIMINATION, AND DECISION TREE ANALYSIS TO SELECT BEST FEATURES FOR IDENTIFYING POIs.
## SELECTORS ARE USED ON 70% OF DATA TO TRY TO PREVENT OVERFITTING


def select_best(data_dict, features_list, n = 10):
	
	### mimicking feature_format but using pandas dataframes instead ###
	
	## create dataframe/array from dictionary
	df = pd.DataFrame.from_dict(data_dict, orient = 'index')
	
	## use feature_list list to trim df
	df = df[features_list]
	## replace NaN string with 0
	df = df.replace('NaN', 0)
	
	## get rid of any rows where all features (not including the target) have 0s
	predictors = list(features_list)
	predictors.remove('poi')
	selection = ~((df[predictors] == 0).all(1)) ## excludes all 0 rows
	df = df[selection]
	## syntax below will not preserve the 'poi' feature, it was a first attempt at removing the zero rows
	#df = df[(df.drop('poi', axis=1)) != 0].dropna(axis=0, how='all').replace(np.nan, 0)

        
        
        ### splitting data into features and labels with test and training sets
        
        ## features and labels
        features_df = df[predictors]
        features = features_df.as_matrix()
        labels_df = df['poi']
        labels = labels_df.as_matrix()
        
        ## test/train split
        ## note that we are using the same random state number here as in gridsearchCV later (in poi_id.py) so that we use the same training data
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        
        
        
        ### getting best features based upon univariate SelectKBest
        kbest = SelectKBest(k = n).fit(features_train, labels_train)
        
        ## using dictionary
        features_scores_dict = {feature: value for (feature, value) in zip(predictors, kbest.scores_)}
        #best_bool = list(kbest.get_support())
        #kbest_selection = [pred for idx, pred in enumerate(predictors) if best_bool[idx]] ## uncomment if selection list below is commented
            
        ## using list for score sorting
        features_scores = sorted(zip(predictors, kbest.scores_), key=lambda x: x[1], reverse = True)
        ## syntax below is faster for larger datasets but requires another library
        #from operator import itemgetter
        #features_scores sorted(zip(predictors, kbest.scores_), key=itemgetter(1), reverse = True)
        kbest_selection = [pair[0] for pair in features_scores[:n]] ## uncomment if 
            
        print "\nBest features from k-best univariate analysis: ", kbest_selection
        print "Feature Scores: ", features_scores
        
        
        
        ### list of features with high correlation
        
        ## getting a list of variable pairs
        pairs = []
        for i in range(0, len(kbest_selection)):
            for j in range(i + 1, len(kbest_selection)):
                pairs.append([kbest_selection[i],kbest_selection[j]])
        
        ## just for fun
        '''pairs = set()
        for i in range(0, len(x)):
            first = x[i]
            for j in range(0, len(x)):
                second = x[j]
                bigger = max(first, second)
                smaller = min(first, second)
                if bigger != smaller:
                    pair = smaller, bigger
                    pairs.add(pair)
        print sorted(pairs)'''
        
        ## list of pairs with correlation coefficients above 0.85
        high_corr = []
        for e in pairs:
            var1_name = e[0]
            var2_name = e[1]
            
            arr1 = df[var1_name].as_matrix()
            arr2 = df[var2_name].as_matrix()
            cor = pearsonr(arr1,arr2)[0]
            e.append(cor) ## append the pair correlation value
            if cor > 0.85:
                high_corr.append(e)
        
        pairs = sorted(pairs, key = lambda x: x[2]) ## sort by correlation value
        
        ## selecting feature in high correlation pair with lowest feature score
        removed = []
        for e in high_corr:
            var1_name = e[0]
            var2_name = e[1]
            var1_score = features_scores_dict[var1_name]
            var2_score = features_scores_dict[var2_name]
            remove = (var1_name if (var1_score < var2_score) else var2_name)
            removed.append(remove)
            
        print "Correlation between feature pairs: ", pairs
        print "Suggested features to remove due to high correlation: ", removed



        ### Recursive feature selection
        clf = RidgeClassifierCV() ## recursive features selection requires linear model
        rfe = RFE(clf, n)
        rfe.fit(features_train, labels_train)
        recursive_selection = [featr for idx, featr in enumerate(predictors) if list(rfe.support_)[idx]]
        print "\nBest features from recursive features selection: ", recursive_selection
        
        
        
        ### Tree-based classifier feature importances
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(features_train, labels_train)
        dt_imp = sorted(zip(clf.feature_importances_, predictors), key=lambda x: x[0], reverse=True)
        dt = [pair[1] for pair in dt_imp[:n]]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        clf.fit(features_train, labels_train)
        rf_imp = sorted(zip(clf.feature_importances_, predictors), key=lambda x: x[0], reverse=True)
        rf = [pair[1] for pair in rf_imp[:n]]
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier()
        clf.fit(features_train, labels_train)
        ab_imp = sorted(zip(clf.feature_importances_, predictors), key=lambda x: x[0], reverse=True)
        ab = [pair[1] for pair in ab_imp[:n]]
        print "\nBest features from Decision Tree: ", dt
        print "Best features from Random Forest: ", rf
        print "Best features from AdaBoost: ", ab
        
        
        
        ### getting best features from all methods
        all_features = kbest_selection + recursive_selection + dt + rf + ab
        unique_features_count = []
        for feat in list(set(all_features)):
            unique_features_count.append([feat, all_features.count(feat)])
            
        unique_features_count = sorted(unique_features_count, key=lambda x: x[1], reverse=True)
        #max_num = max(unique_features_count, key=lambda x: x[1])[1] ## if we want to keep based upon some frequency number
        keepers = [ft[0] for ft in unique_features_count[0:n]]
        print "\nBest features from all selection methods: ", keepers
        
        
        return keepers


        
if __name__ == "__main__":
    
    import pickle
    enron_data = pickle.load(open("final_project_dataset.pkl", "r"))
    enron_data.pop('TOTAL', 0)
    
    practice_features = ['poi', 'salary', 'deferral_payments', 'total_payments',
    'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
    'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
    'long_term_incentive', 'restricted_stock', 'director_fees']
    
    print select_best(enron_data, practice_features, n = 10)
    
    
