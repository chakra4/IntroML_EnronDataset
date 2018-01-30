#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from datetime import datetime

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if (poi_messages == 'NaN' or all_messages == 'NaN'):
      return fraction;
    if (all_messages == 0) :
      return fraction;
      
    fraction = float(poi_messages) / float(all_messages)
    return fraction

def GaussianNBPipeline():
  estimators = [("scaling", MinMaxScaler()), ('reduce_dim', PCA()), ('clf', GaussianNB())]
  pipe = Pipeline(estimators)
  params = dict(reduce_dim__n_components=[5, 7, 9, 11, 13])
  #specify the number of splits (in this example it will be  100):
  cross_val = StratifiedShuffleSplit(100, test_size = 0.3, random_state = 42)
  clf = GridSearchCV(pipe, param_grid=params, scoring='f1', cv =cross_val)
  return clf

def GaussianNBPipelineSelectKBest():
  estimators = [("scaling", MinMaxScaler()), ('reduce_dim', SelectKBest()), ('clf', GaussianNB())]
  pipe = Pipeline(estimators)
  #params = dict(reduce_dim__score_func=[f_classif],reduce_dim__k=[5, 7, 9, 11, 13])
  params = dict(reduce_dim__k=[5, 7, 9, 11, 13])
  #specify the number of splits (in this example it will be  100):
  cross_val = StratifiedShuffleSplit(50, test_size = 0.3, random_state = 42)
  clf = GridSearchCV(pipe, param_grid=params, scoring='f1', cv =cross_val)
  return clf

def DecisionTreePipeline():
  estimators = [("scaling", MinMaxScaler()), ('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]
  pipe = Pipeline(estimators)
  params = dict(reduce_dim__n_components=[5,7,9], \
                clf__max_depth=[None,3,5,7,9],clf__min_samples_split=[2,3,4])
  #specify the number of splits (in this example it will be 20):
  cross_val = StratifiedShuffleSplit(20, test_size = 0.3, random_state = 42)
  clf = GridSearchCV(pipe, param_grid=params, scoring='f1', cv =cross_val)
  return clf

def SVCPipeline():
  estimators = [("scaling", MinMaxScaler()),('reduce_dim', PCA()), ('clf', SVC())]
  pipe = Pipeline(estimators)
  params = dict(reduce_dim__n_components=[5, 7, 9, 11, 13],
                clf__kernel=['rbf'],
                clf__gamma=[0.0005, 0.005, 0.05, 0.1, 0.5],
                clf__C=[10000, 5000, 1000, 100, 10])
  cross_val = StratifiedShuffleSplit(20, test_size = 0.3, random_state = 42)
  clf = GridSearchCV(pipe, param_grid=params, scoring='f1', cv =cross_val)
  return clf

def AdaBoostPipeline():
  estimators = [("scaling", MinMaxScaler()),('reduce_dim', PCA()), ('clf', AdaBoostClassifier())]
  pipe = Pipeline(estimators)
  params = dict(reduce_dim__n_components=[5,7,9,11],clf__n_estimators=[50], clf__learning_rate=[1.0])
  cross_val = StratifiedShuffleSplit(10, test_size = 0.3, random_state = 42)
  clf = GridSearchCV(pipe, param_grid=params, scoring='f1', cv =cross_val)
  return clf

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees', \
'fraction_from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop( 'TOTAL', 0 ) 

### Task 3: Create new feature(s)
for key in data_dict:
  data_dict[key]['fraction_from_this_person_to_poi'] = \
      computeFraction(data_dict[key]['from_this_person_to_poi'], data_dict[key]['from_messages'])
  #print data_dict[key], data_dict[key]['fraction_from_this_person_to_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf = GaussianNBPipelineSelectKBest()
#clf = GaussianNBPipeline()
#clf = DecisionTreePipeline()
#clf = SVCPipeline()
#clf = AdaBoostPipeline()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#Testing number of PCA
#min_max_scaler = MinMaxScaler()
#scaled_features_train = min_max_scaler.fit_transform(features_train)
#pca = PCA(n_components=10, whiten=True, svd_solver='randomized').fit(scaled_features_train)
#print 'PCA VARIANCE: ' , pca.explained_variance_ratio_

#clf = clf.fit(features_train, labels_train)
# Use all of the data
clf = clf.fit(features, labels)

print "Classifier Score: ", clf.score(features_test, labels_test)
pred = clf.predict(features_test)

print "Lables_Test: ", labels_test
print "Pred: " , pred

from sklearn.metrics import recall_score
print "Recall Score:", recall_score(labels_test, pred)

from sklearn.metrics import precision_score
print "Precision Score:", precision_score(labels_test, pred)

from sklearn.metrics import f1_score
print 'F1 Score:',f1_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print "Best params: ", clf.best_params_
dump_classifier_and_data(clf, my_dataset, features_list)
