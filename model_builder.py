# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
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

# this function runs grid search on several parameters
def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):
    ps = paramsearch(params)
    # search 'border_count', 'l2_leaf_reg' etc. individually 
    #   but 'iterations','learning_rate' together
    for prms in chain(ps.grid_search(['border_count']),
                      ps.grid_search(['ctr_border_count']),
                      ps.grid_search(['l2_leaf_reg']),
                      ps.grid_search(['iterations','learning_rate']),
                      ps.grid_search(['depth'])):
        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
        # save the crossvalidation result so that future iterations can reuse the best parameters
        ps.register_result(res,prms)
        print(res,prms,'best:',ps.bestscore(),ps.bestparam())
        
    return ps.bestparam()


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

train = train.fillna(999)
x_train = x_train.fillna(999)

print('training catboost..')

params_old = {'iterations': 2,
          'learning_rate': 1,
          'depth': 2,
          'custom_loss': 'AUC'}

params_new = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200],
          'ctr_border_count':[50,5,10,20,100,200],
          'thread_count':4,
          'custom_loss': 'AUC'}


bestparams = catboost_param_tune(params_new, x_train,
                                 y_train, cat_cols_idx)


# train classifier with tuned parameters    
clf = cb.CatBoostClassifier(**bestparams)
clf.fit(x_train, np.ravel(y_train), cat_features=cat_cols_idx)

# How do I want to do this last section? Just a random cut of data?
# print('error:',1-np.mean(res==np.ravel(test_label)))

print('predicting model outputs..')
y_pred = fit_model.predict_proba(x_test)[:,1]

pred_valid = model.predict(valid_pool)

#I'd prefer to do K-fold here with n = 5 to get the better understanding of the error
gini_norm_valid = round(gini_normalized(y_valid, pred_valid), 4)

print('gini (normalised) for the validation set %s' % (gini_norm_valid))

os.chdir(sub_dir)

# Create a submission file
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = y_pred
sub.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), gini_norm_valid),
                         index=False, float_format='%.4f')

