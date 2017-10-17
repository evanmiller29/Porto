# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
import xgboost as xgb
from datetime import datetime

#==============================================================================
# Defining loss function
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
 
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

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
ind_cols = [i for e in ['ind'] for i in train_cols_up  if e in i]
reg_cols = [i for e in ['reg'] for i in train_cols_up  if e in i]
car_cols = [i for e in ['car'] for i in train_cols_up  if e in i]

cat_cols_idx = [i for i, c in enumerate(train_cols_up) if 'cat' in c]

ttl_used = binary_cols + cat_cols + ind_cols + reg_cols + car_cols
calc_cols = [col for col in train_cols_up if col not in ttl_used]

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

x_train = x_train.fillna(999)
x_valid = x_valid.fillna(999)

d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(x_test)

# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'dart'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 6
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Train the model! We pass in a max of 10,000 rounds (with early stopping after 100)
# and the custom metric (maximize=True tells xgb that higher metric is better)

mdl = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100, 
                feval=gini_xgb, maximize=True, verbose_eval=100)

pred_valid = mdl.predict(d_valid)
gini_norm_valid = round(gini_normalized(y_valid, pred_valid), 4)

print('gini (normalised) for the validation set %s' % (gini_norm_valid))

p_test = mdl.predict(d_test)

print('Creating a submission file..')
os.chdir(sub_dir)

sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = p_test

sub.to_csv(sample.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), gini_norm_valid),
                         index=False, float_format='%.4f'), index=False)