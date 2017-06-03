#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'deferred_income',
                 'deferral_payments',
#                 'loan_advances',
                 'other',
                 'expenses',
                 'director_fees',
                 'total_payments',

                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 
                 'stock_payment_ratio',
                 'salary_payment_ratio',
                 'bonus_payment_ratio',
                 
#                 'empty_count',

#                 'email_address',
                 'shared_receipt_with_poi',
                 'from_messages',
                 'from_this_person_to_poi',
                 'to_messages',
                 'from_poi_to_this_person',
                 
                 'poi_contact_ratio'
                 ]

### Load the dictionary containing the dataset

print "Loading Data"

with open("final_project_dataset.pkl", "r") as data_file:
    
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

_ =  data_dict.pop('TOTAL')

outliers=[
        'LAY KENNETH L',
        'MARTIN AMANDA K',
        'BHATNAGAR SANJAY',
        'FREVERT MARK A'
        ]


for item in outliers:
    _ = data_dict.pop(item)


def count_empty(entry):
    ct = 0
    for key in entry:
        if entry[key] == 'NaN':
            ct+=1
    return ct

poor_entries = []
for person in data_dict:
    data_dict[person]['empty_count'] = count_empty(data_dict[person])
    if count_empty(data_dict[person]) > 17:
        poor_entries.append(person)

#for item in poor_entries:
#    _ = data_dict.pop(item)



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

print "Total Data Points: ", len(my_dataset)


def formatnan(number):
    if np.isnan(number):
        return 'NaN'
    else:
        return number



print "Creating new features..."
for person in my_dataset:
       
    
    my_dataset[person]['empty_count']=count_empty(my_dataset[person])
    
    ttpmt=np.float(my_dataset[person]['total_payments'])
    ttstk=np.float(my_dataset[person]['total_stock_value'])
    slry=np.float(my_dataset[person]['salary'])
    bnus=np.float(my_dataset[person]['bonus'])
    
    my_dataset[person]['stock_payment_ratio'] = formatnan(ttstk/ttpmt)
    my_dataset[person]['salary_payment_ratio'] = formatnan(slry/ttpmt)
    my_dataset[person]['bonus_payment_ratio'] = formatnan(bnus/ttpmt)
      
    topoi=np.float(my_dataset[person]['from_this_person_to_poi'])
    frompoi=np.float(my_dataset[person]['from_poi_to_this_person'])
    tomsg=np.float(my_dataset[person]['to_messages'])
    frommsg=np.float(my_dataset[person]['from_messages'])
    
    my_dataset[person]['poi_contact_ratio'] = \
              formatnan((topoi+frompoi)/(tomsg+frommsg))


### Extract features and labels from dataset for local testing
print "Extracting features..."
data = featureFormat(my_dataset, features_list, remove_NaN=True, 
                  remove_all_zeroes=True, remove_any_zeroes=False, 
                  sort_keys = True)
labels, features = targetFeatureSplit(data)

### Sub-routine to explore the data, turned False in final
if False:
    
    print "Data exploration"

    featnp = np.array(data)
    featdf = pd.DataFrame(data, columns=features_list)
    
    feat_basic = featdf.describe()
    feat_stat = featdf.groupby('poi').count()

    #for feat in features_list[1:]:
    #    qt=featdf[feat].quantile(1)
    #    featdf[['poi', feat]][featdf[feat]<=qt].boxplot(by='poi')
    #    plt.show()

    for feat in features_list[1:]:
        df=pd.DataFrame()
        df['poi']=featdf['poi']
        df['feat_cut']=pd.cut(featdf[feat], 20)
        df1 = pd.value_counts(df[df['poi']==0.0]['feat_cut'], sort=False)
        df2 = pd.value_counts(df[df['poi']==1.0]['feat_cut'], sort=False)
        dfp= pd.concat([df1, df2], axis=1)
        dfp.columns=['non poi', 'poi']
        dfp.plot(kind='bar', title=feat)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree, naive_bayes, svm, ensemble, neighbors
from sklearn import decomposition, feature_selection, model_selection, preprocessing
from sklearn import pipeline

print "Setting up classifiers and GridSearchCV"

pca = decomposition.PCA(10)
fselect = feature_selection.SelectKBest(k=10)

mmx = preprocessing.MinMaxScaler()
std = preprocessing.StandardScaler()
rbs = preprocessing.RobustScaler(quantile_range=(0.0, 90.0))


det = tree.DecisionTreeClassifier()
gnb = naive_bayes.GaussianNB()
adb = ensemble.AdaBoostClassifier()
knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
svc = svm.SVC()



pipe = pipeline.Pipeline([
        ('scaler', None),
        ('feat_select', fselect),
        ('core_clf', gnb)        
        ])


# Params1 to evaluate which algorith, scaler, and feature selection combination works best
params1 = {
#        'scaler': [None, mmx, rbs, std],
        'feat_select': [fselect, pca],
        'core_clf': [det, gnb, adb, knn, svc]
        }

# Params2 to optimize the SelectKBest parameter when GaussianNB classifier
params2 = {
        'scaler': [None],
        'feat_select': [fselect],
        'feat_select__k': [2, 3, 5, 8, 10, 12],
        'core_clf': [gnb]
        }

# Params3 to evaluate the decision tree parameter to optimize result
params3 = {
        'scaler': [None],
        'feat_select': [pca],
        'feat_select__n_components': [10],
        'core_clf': [det],
        'core_clf__min_samples_split': [2, 3, 5, 8, 10, 12, 15]
        }


# NOTE: Choose from params1, params2, and params3 to perform different 
# evaluation tasks.

#cvgen = model_selection.StratifiedShuffleSplit(n_splits=3)
cvgen = model_selection.StratifiedKFold(n_splits=3)

gscv = model_selection.GridSearchCV(
        pipe, 
        param_grid=params1, 
        scoring = 'f1',
        cv=cvgen)

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

print "POI: train: ", sum(labels_train), ", test: ", sum(labels_test)

print "Training and selection of classifiers"
gscv.fit(features, labels)

clf = gscv.best_estimator_
cvresults = pd.DataFrame(gscv.cv_results_)


if clf.get_params()['feat_select']==fselect:
    feat_selected = clf.get_params()['feat_select'].get_support()
    feat_scores = clf.get_params()['feat_select'].scores_
    feat_names = features_list[1:]
    K_best_features = pd.DataFrame([feat_selected, feat_scores, feat_names])
    print K_best_features.transpose().sort_values(by=1, ascending=False)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print "Training complete. Best classifier dumping to harddrive"

from tester import test_classifier
test_classifier(clf, my_dataset, features_list)


dump_classifier_and_data(clf, my_dataset, features_list)


