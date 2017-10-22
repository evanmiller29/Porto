# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime
from itertools import product,chain

#==============================================================================
# Defining custom functions
#==============================================================================

def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
 
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# this function does 3-fold crossvalidation with catboostclassifier          
def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):
    kf = KFold(n_splits=n_splits,shuffle=True) 
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index,:]
        test = train_set.iloc[test_index,:]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = cb.CatBoostClassifier(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))
    return np.mean(res)

#==============================================================================
# Reading in data
#==============================================================================

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Porto'
sub_dir = 'F:/Nerdy Stuff/Kaggle submissions/Porto'
script_dir = 'C:/Users/Evan/Documents/GitHub/Porto'

os.chdir(script_dir)

from paramsearch import paramsearch

os.chdir(base_dir)

train = pd.read_csv('train.csv', low_memory = True)
test = pd.read_csv('test.csv', low_memory = True)
sample = pd.read_csv('sample_submission.csv', low_memory = True)

train_columns = list(train.columns.values)
test_columns = list(test.columns.values)

print('Base cols not in either set: %s' % (list(set(train_columns) - set(test_columns))))

y_train = train['target']
train_id = train['id']
x_train = train.drop(['target', 'id'], axis = 1)

y_test = []
test_id = test['id']
x_test = test.drop('id', axis = 1)

train_cols_up = x_train.columns.values

binary_cols = [i for e in ['bin'] for i in train_cols_up  if e in i]
cat_cols = [i for e in ['cat'] for i in train_cols_up  if e in i]

cat_cols_idx = [i for i, c in enumerate(train_cols_up) if 'cat' in c]

x_train = x_train.fillna(999)

X = x_train.values
y = y_train.values

print('training catboost..')

params = {'depth':2,
          'iterations':100,
          'learning_rate':0.001, 
          'l2_leaf_reg':5,
          'border_count':10,
          'ctr_border_count':50,
          'thread_count':5,
          'custom_loss': 'Logloss'}

kfold = 3
skf = StratifiedKFold(n_splits=kfold, random_state=42)

os.chdir(sub_dir)

sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)

gini_res = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # Convert our data into Catboost format
    
    train_pool = cb.Pool(X_train, y_train, cat_cols_idx)
    valid_pool = cb.Pool(X_valid, y_valid, cat_cols_idx)
    
    clf = cb.CatBoostClassifier(**params)
    clf.fit(train_pool)
    valid_preds = clf.predict(valid_pool)
    
    gini_norm_valid = round(gini_normalized(y_valid, valid_preds), 4)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    print('Accuracy on validation set: %s' % (gini_norm_valid))
    
    print('predicting model outputs..')
    
    p_test = clf.predict_proba(x_test)[:, 1]
    sub['target'] += p_test/kfold

    gini_res.append(gini_norm_valid)

# Create a submission file

avg_gini = str(round(np.mean(gini_res), 4))[0:10]

sub.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), avg_gini),
                         index=False, float_format='%.4f')

