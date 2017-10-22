# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os
import numpy as np
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from numba import jit

#==============================================================================
# Defining custom functions
#==============================================================================

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

#==============================================================================
# Reading in data
#==============================================================================

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Porto'
sub_dir = 'F:/Nerdy Stuff/Kaggle submissions/Porto'
script_dir = 'C:/Users/Evan/Documents/GitHub/Porto'

os.chdir(base_dir)

train = pd.read_csv('train.csv', low_memory = True)
test = pd.read_csv('test.csv', low_memory = True)
sample = pd.read_csv('sample_submission.csv', low_memory = True)

# =============================================================================
# Splitting dataframes
# =============================================================================

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

# =============================================================================
# Formatting data for catboost
# =============================================================================

x_train = x_train.fillna(999)
x_test = x_test.fillna(999)

X = x_train.values
y = y_train.values

print('training catboost..')

modelMeta = dict()
modelMeta['MAX_ROUNDS'] = 650
modelMeta['OPTIMIZE_ROUNDS'] = False
modelMeta['LEARNING_RATE'] = 0.05

params = {'learning_rate': modelMeta['LEARNING_RATE'], 
            'depth': 6, 
            'l2_leaf_reg' : 8, 
            'iterations' : modelMeta['MAX_ROUNDS'],
            'border_count':10,
            'ctr_border_count':50,
            'loss_function' : 'Logloss'}

kfold = 3
skf = StratifiedKFold(n_splits=kfold, random_state=42)

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
    
    if modelMeta['OPTIMIZE_ROUNDS']:
        
        fit_model = clf.fit(train_pool,
                            eval_set = valid_pool,
                            use_best_model = True
                            )
        
        print( "  N trees = ", clf.tree_count_ )
        
    else:
        
        fit_model = clf.fit(train_pool)
    
    valid_preds = fit_model.predict(valid_pool)
    
    gini_norm_valid = round(eval_gini(y_valid, valid_preds), 4)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    print('Accuracy on validation set: %s' % (gini_norm_valid))
    
    print('predicting model outputs..')
    
    p_test = fit_model.predict_proba(x_test)[:, 1]
    sub['target'] += p_test/kfold

    gini_res.append(gini_norm_valid)

# =============================================================================
# Outputting results to a submission file
# =============================================================================

avg_gini = str(round(np.mean(gini_res), 4))[0:10]

os.chdir(sub_dir)

sub.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), avg_gini),
                         index=False, float_format='%.4f')

