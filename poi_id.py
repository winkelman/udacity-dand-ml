#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("./tools/") # DIFFERENT GITHUB PATH HERE

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data







### Task 1: Preliminary Feature Selection

financial_features = ['salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees'] ## all units are in US dollars

email_features = ['to_messages', 'from_poi_to_this_person', 'email_address',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] ## units are number of email messages, 'email_address' is a string

poi_label = ['poi'] ## boolean

email_features.remove('email_address') ## not numerical

## features_list is a list of strings, the first feature must be "poi".
features_list = poi_label + financial_features #+ email_features ## email_features excluded because of data leakage


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )







### Task 2: Data Exploration and Outlier Removal (not enough data for 10% rule)

### Create Pandas Dataframe
import pandas as pd
import numpy as np
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
#print df.isnull().any() ## our NaN values are actually strings
## we can replace string NaNs with 0 or real NaN
#df = df.replace('NaN', 0)
df = df.replace('NaN', np.nan) ## np.nan works better for data summary stats and plotting


### Data Summary
print "\n\nData Inspection:\n\n", df.info()
print "\n\n\nData Summary:\n\n", df.describe()

## non-null counts for POIs
print "\n\n\nPOI Data Inspection:\n\n", df[df.poi == True].info()


### Visual Inspection/Removal of Outliers

## from looking at the enron insiderpay.pdf file
print "\n\nFirst outlier removed is: 'THE TRAVEL AGENCY IN THE PARK'"
df = df.drop('THE TRAVEL AGENCY IN THE PARK')
#data_dict.pop('THE TRAVEL AGENCY IN THE PARK') ## not necessary here, can convert back to dict later

import seaborn as sns
## if NaN strings are kept, we need to create a new df that screens these values for the plot
#df_plt = df[(df.total_payments != "NaN") & (df.total_stock_value != "NaN")]

## the pair plot below would be ideal (incorrect syntax) to detect outliers among features...
#sns.pairplot(df, x_vars=['total_stock_value', 'total_payments'], y_vars=['total_stock_value', 'total_payments'], kind="scatter", dropna=True)
## instead of pair plot (cannot render), using 'total_payments' and 'total_stock_value' features to detect outliers
sns.jointplot(x="total_payments", y="total_stock_value", data=df, dropna=True)
sns.plt.show()

## the plot shows one clear outlier which we also found in lesson 7, 'TOTAL'
out2name = df['total_payments'].idxmax()
print "\nSecond outlier removed is:", out2name 
df = df.drop('TOTAL')
#data_dict.pop('TOTAL') ## again, not necessary here

## check if any more outliers
sns.jointplot(x="total_payments", y="total_stock_value", data=df, dropna=True)
sns.plt.show()  ## one more clear outlier

## second outlier, 'LAY KENNETH L'
out3name = df['total_payments'].idxmax()
print "\nThird outlier NOT removed is:", out3name
#data_dict.pop(out3name) ## again, not necessary here
#df = df.drop(out3name) ## we have too few POIs/data to remove this anyway

## remove people(s) with no financial data
no_data = df[df[financial_features].isnull().all(axis = 1)].index.values ## only 1
print "\nFourth person removed with no data:", no_data
df = df.drop(no_data)

## this would remove all people with no total payments and no total stock value (2 more than necessary)
'''
no_data = df[np.isnan(df.total_payments) & np.isnan(df.total_stock_value)].index.values.tolist()
print "\nPeople with no data removed from dataset: ", no_data, "\n"
for person in no_data:
    df = df.drop(person)
    #data_dict.pop(person)
'''


### Convert DataFrame Back to Dictionary
df = df.replace(np.nan, 'NaN') ## convert np.nan back to 'NaN' for 'add_words' in Task 3
#data_dict = df.to_dict(orient = 'index') ## this doesn't work, 'index' is deprecated?
df = df.transpose()
data_dict = df.to_dict()


### Inspect Test/Train Split
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
print "\n", len(labels_train), "total people in the TRAINING set with", sum(labels_train), "POIs at", 100.0*sum(labels_train)/len(labels_train), "percent."
print len(labels_test), "total people in the TEST set with", sum(labels_test), "POIs at", 100.0*sum(labels_test)/len(labels_test), "percent."







### Task 3: Feature Engineering and Feature Selection

### Prune Financial Features based upon Data Exploration (# of POIs and missing data)

df = df.transpose()[features_list] ## using same dataframe from the outlier removal section
df = df.replace('NaN', np.nan)

## creating list of important features by number of non-null values
feature_data = []
alpha = 1.0* sum(df.poi != True) / sum(df.poi == True) ## weighting POIs by their proportion in the data set

for ftr in list(df.columns.values):
    if ftr not in ['poi']:
        n_poi = sum( df[df.poi == True][ftr].notnull() )
        n_non_poi = sum( df[df.poi != True][ftr].notnull() )
        feature_data.append([ftr, (alpha*n_poi + n_non_poi)])
        
feature_data = sorted(feature_data, key=lambda x: x[1], reverse = True)
new_features = [ftr[0] for ftr in feature_data[0:8]] ## top 8
features_list = ['poi'] + new_features ## take top 8 and add back POI

## remove new people with no data in updated features
no_data = df[df[new_features].isnull().all(axis = 1)].index.values
no_data = ''.join(no_data) ## only 1
data_dict.pop(no_data)
    
print "\n\n\nTop 8 features with most data:", new_features ## exclude POI in the printed list
print "\nExcluding these features because of limited data:", [ftr for ftr in list(df.columns.values) if ftr not in features_list]
print "\nRemoved 1 additional person with no data in new features:", no_data, "\n\n\n"



### Creating Suspicious Words Features

## adding email text data for each person
from add_email_words import add_words ## adds all email text data as one string for each person in data_dict[person]['words']
print "Processing emails...\n"
data_dict = add_words(data_dict, all=False, n=100) ## 'all' to process entire email corpus, n is number of emails per person, default is 30

## getting list of suspicious words
from get_important_words import get_words
print "\nGetting important words..."
impt_words = get_words(data_dict)
# UNCOMMENT BELOW WITH ACCESS TO FULL EMAIL CORPUS
'''if u'boardroom' not in impt_words: ## have to add 1 critically important word manually, the DT classifier doesn't always find it for some reason...?
    #impt_words.append(u'boardroom')'''
    
print "\nSuspicious words are:", impt_words

## add count of each suspicious word for all persons to dictionary in data_dict[person]['a_suspicious_word_here']
from add_word_count import add_count
print "\nAdding suspicious word counts..."
data_dict = add_count(data_dict, impt_words)

## removing 'word' feature from data_dict, no longer needed
for person in data_dict:
    data_dict[person].pop("words")

## inspecting suspicious words count for POIs
'''
for person in data_dict:
    poi = data_dict[person]['poi']
    for word in impt_words:
        num = data_dict[person][word]
        if num > 0:
            print poi, person, word, num 
'''
## rename
word_features = impt_words

## add to word features to feature list
features_list += word_features



### Selecting Best Features (THIS IS IMPLEMENTED HERE AS IT CANNOT BE DONE IN A PIPELINE)

from feature_selector import select_best
print "\n\n\nGetting best features..."
best_features = select_best(data_dict, features_list, 6)

final_features = best_features
## the best 6 aren't always the same here, input manually to be sure
# UNCOMMENT BELOW WITH ACCESS TO FULL EMAIL CORPUS
#final_features = ['total_stock_value', u'boardroom', 'bonus', u'blown', 'exercised_stock_options', 'restricted_stock'] #, 'salary']
print "\nFinal Features Used: ", final_features, "\n"


### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi'] + final_features


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)







### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()
from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier()
from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier()
from sklearn.cluster import KMeans ## To do...
#clf = KMeans(n_clusters=2)


### Scaling and PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

#estimators = [('scaler', StandardScaler()), ('reducer', PCA()), ('ada', AdaBoostClassifier())]  ## standard scaling and PCA vastly improve AdaBoost
estimators = [('scaler', MinMaxScaler()), ('knn',  KNeighborsClassifier())] ## KNN requires range scaling but PCA offers no benefit


### Classifier Selection
#clf = GaussianNB()  ## this is the benchmark 
clf = Pipeline(estimators)


### Parameter Optimization
#from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## standard scaler, pca, adaboost
'''parameters = dict(ada__base_estimator = [DecisionTreeClassifier(), DecisionTreeClassifier(max_depth = 5), DecisionTreeClassifier(max_depth = 8)],
                ada__learning_rate = [1.0, 5.0, 10.0, 20.0])'''
## min-max scaler, knn
parameters = dict(knn__n_neighbors = range(1, 6), knn__weights = ['uniform', 'distance'])

sss = StratifiedShuffleSplit(labels_train, random_state = 11) ## best cross-validation method for small data set and few POIs
clf = GridSearchCV(clf, param_grid = parameters, scoring = 'recall', cv = sss, verbose = 1)  ## maximizing the f1 score, recall results in same

## grid fit for best estimator
clf.fit(features_train, labels_train)

## assign classifier to best estimator
clf = clf.best_estimator_
print "\nBest parameters: \n", clf.get_params()







### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## convert features and labels to arrays
#print type(labels), type(features)
features = np.asarray(features)
labels = np.asarray(labels)

## lists of metrics
accuracy = []
precision = []
recall = []
f_one = []

## or for classification report
total_labels = np.array([])
total_preds = np.array([])

folds = 200
sss = StratifiedShuffleSplit(labels, folds, test_size = 0.3, random_state = 11)

for train_idx, test_idx in sss:
    features_train, labels_train = features[train_idx], labels[train_idx]
    features_test, labels_test = features[test_idx], labels[test_idx]
    
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    #pred = clf.fit_predict(features) ## k-means
    
    #acc = clf.score(features_test, labels_test)
    acc = accuracy_score(labels_test, pred)
    pre = precision_score(labels_test, pred)
    rec = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)
    
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f_one.append(f1)
    
    total_labels = np.concatenate((total_labels, labels_test))
    total_preds = np.concatenate((total_preds, pred))

## not quite the same as tester.py
print "\nAccuracy: ", sum(accuracy)/folds, "\nPrecision: ", sum(precision)/folds, "\nRecall: ", sum(recall)/folds, "\nF1-Score: ", sum(f_one)/folds

## using sklearn
from sklearn.metrics import classification_report
## precision, recall, f1-score, and support (number of occurrences in test set)
print "\n", classification_report(total_labels, total_preds, target_names=['non-POI', 'POI']) ## these scores appear to be less accurate...







### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

