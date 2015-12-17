#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data






### Task 1: Select what features you'll use.

financial_features = ['salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees']
## all units are in US dollars

email_features = ['to_messages', 'from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
## units are number of email messages
## 'email_address', a text string, is not included

poi_label = ['poi'] ## boolean

preliminary_features = poi_label + financial_features #+ email_features ## email_features excluded because of data leakage

## features_list is a list of strings, each of which is a feature name.
## the first feature must be "poi".
features_list = list(preliminary_features)


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )







### Task 2: Remove outliers (not enough data for 10% rule)

### Create Pandas Dataframe
import pandas as pd
import numpy as np
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
#print df.isnull().any() ## our NaN values are actually strings
## we can replace string NaNs with 0 or actual NaN
#df = df.replace('NaN', 0)
df = df.replace('NaN', np.nan) ## np.nan works better for data summary stats and plotting


### Data Summary
print "\n", df.info()
print "\n", df.describe()


### Visual Inspection/Removal of Outliers
import seaborn as sns
## if NaN strings are kept, we need to create a new df that screens these values for the plot
#df_plt = df[(df.total_payments != "NaN") & (df.total_stock_value != "NaN")]

## the pair plot below would be ideal (incorrect syntax) to detect outliers among features...
#sns.pairplot(df, x_vars=['total_stock_value', 'total_payments'], y_vars=['total_stock_value', 'total_payments'], kind="scatter", dropna=True)
## instead of pair plot (cannot render), using 'total_payments' and 'total_stock_value' featuers to detect outliers
sns.jointplot(x="total_payments", y="total_stock_value", data=df, dropna=True)
sns.plt.show()

## this plot shows one clear outlier which we also found in lesson 7, 'TOTAL'
#data_dict.pop('TOTAL') ## not necessary, can convert df to dict when finished
df = df.drop('TOTAL')

## any more outliers?
sns.jointplot(x="total_payments", y="total_stock_value", data=df, dropna=True)
sns.plt.show()  ## one more clear outlier

## second outlier, 'LAY KENNETH L'
out2name = df['total_payments'].idxmax()
#data_dict.pop(out2name) ## again, not necessary here
#df = df.drop(out2name) ## we have too few POIs/data to remove this anyway

## remove people with essentially no data (no total_payments and no total_stock_value)
no_data = df[np.isnan(df.total_payments) & np.isnan(df.total_stock_value)].index.values.tolist()
print "\nPeople with no data removed from dataset: ", no_data, "\n"
for person in no_data:
    df.drop(person)
    
## from looking at the enron insider pay pdf file
df.drop('THE TRAVEL AGENCY IN THE PARK')


### Convert DF Back to Dictionary
df = df.replace(np.nan, 'NaN') ## convert np.nan back to 'NaN' for 'add_words' in Task 3
#data_dict = df.to_dict(orient = 'index') ## this doesn't work, 'index' is deprecated?
df = df.transpose()
data_dict = df.to_dict()







### Task 3: Create new feature(s)

### Creating Suspicious Words Features
from add_email_words import add_words ## adds all email text data as one string for each person in data_dict[person]['words']
data_dict = add_words(data_dict, n=2) ## n is number of emails per person, default is 30, must be 2 or greater

## getting list of suspicious words
from get_important_words import get_words
impt_words = get_words(data_dict)
print "Suspicious words are: ", impt_words

## add count of each suspicious word for all persons to dictionary in data_dict[person]['a_suspicious_word_here']
from add_word_count import add_count
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


### Updating Features List
word_features = impt_words
#preliminary_features = preliminary_features + word_features ## word_features did not improve predictions but complicated feature selection algorithms



### Task 3.1: Selecting Best Features
# THIS IS IMPLEMENTED HERE AS IT CANNOT BE DONE IN A PIPELINE

## Updating features list based upon available data
df = df.transpose()[features_list] ## using same pandas data frame from the outlier removal section
df = df.replace('NaN', np.nan)
print "\n", df[df.poi == True].info() ## non-null counts for POIs
feature_data = [] ## list of important features by number of non-null values
alpha = 1.0* sum(df.poi != True) / sum(df.poi == True) #/ sum(df.poi != True) ## weighting POIs by their proportion in the data set
for ftr in list(df.columns.values):
    if ftr not in ['poi']:
        n_poi = sum( df[df.poi == True][ftr].notnull() )
        n_non_poi = sum( df[df.poi != True][ftr].notnull() )
        feature_data.append([ftr, (n_poi + alpha*n_non_poi)])
        
feature_data = sorted(feature_data, key=lambda x: x[1], reverse = True)
features_list = [ftr[0] for ftr in feature_data[0:8]] + ['poi'] ## take top 8 and add back POI
print "\nTrimming features to include top 8 with most data: ", features_list[:-1] ## excluse POI in the printed list
print "Excluding these features because of limited data: ", [ftr for ftr in list(df.columns.values) if ftr not in features_list]

## Selecting best features based upon several selection methods
from feature_selector import select_best
updated_features = select_best(data_dict, features_list, 5)



### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi'] + updated_features ## features selected using k-best, recursive feature elimination, and tree based classifiers
## manual pruning from worst correlated features and rankings...
if 'expenses' in features_list:
    features_list.remove('expenses')
if 'salary' in features_list:
    features_list.remove('salary')
#features_list.remove('exercised_stock_options')
print "\nFinal Features Used: ", features_list, "\n"


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
from sklearn.cross_validation import train_test_split
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

