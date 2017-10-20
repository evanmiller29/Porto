# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from catboost import CatBoostClassifier, Pool
from datetime import datetime

#==============================================================================
# Defining loss function
#==============================================================================


def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_catboost(pred, y):
    return gini(y, pred) / gini(y, y)

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

#==============================================================================
# Reading in data
#==============================================================================

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Porto'
sub_dir = 'F:/Nerdy Stuff/Kaggle submissions/Porto'

os.chdir(base_dir)

train = pd.read_csv('train.csv', low_memory = True)
test = pd.read_csv('test.csv', low_memory = True)
sample = pd.read_csv('sample_submission.csv', low_memory = True)

train_columns = list(train.columns.values)
test_columns = list(test.columns.values)

print('Base cols not in either set: %s' % (list(set(train_columns) - set(test_columns))))

y_train = train['target'].values
train_id = train['id'].values
x_train = train.drop(['target', 'id'], axis = 1)

y_test = []
test_id = test['id'].values
x_test = test.drop('id', axis = 1)

train_cols_up = x_train.columns.values

binary_cols = [i for e in ['bin'] for i in train_cols_up  if e in i]
cat_cols = [i for e in ['cat'] for i in train_cols_up  if e in i]

cat_cols_idx = [i for i, c in enumerate(train_cols_up) if 'cat' in c]

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

x_train = x_train.fillna(999)
x_valid = x_valid.fillna(999)

print('training catboost..')

train_pool = Pool(x_train, y_train, cat_features=cat_cols_idx)
valid_pool = Pool(x_valid, y_valid, cat_features=cat_cols_idx)

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

params_set = [params_old, params_new]

model = CatBoostClassifier(**params_old)
fit_model = model.fit(train_pool)

print('predicting model outputs..')
y_pred = fit_model.predict_proba(x_test)[:,1]

pred_valid = model.predict(valid_pool)
gini_norm_valid = round(gini_normalized(y_valid, pred_valid), 4)

print('gini (normalised) for the validation set %s' % (gini_norm_valid))

os.chdir(sub_dir)

# Create a submission file
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = y_pred
sub.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), gini_norm_valid),
                         index=False, float_format='%.4f')

