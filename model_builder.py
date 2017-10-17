# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os
import numpy as np
from gini import gini_normalized
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from catboost import CatBoostClassifier, Pool
from datetime import datetime

gini_sklearn = make_scorer(gini_normalized, True, True)

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Porto'
sub_dir = 'F:/Nerdy Stuff/Kaggle/Porto/Submissions'

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

train_pool = Pool(data = x_train, label = y_train, cat_features = cat_cols_idx)
valid_pool = Pool(data = x_valid, label = y_valid, cat_features = cat_cols_idx)

model = CatBoostClassifier(iterations=10, learning_rate=0.01, verbose=True, custom_loss = gini_sklearn)
model.fit(train_pool)

pred_valid = model.predict_proba(x_valid)[:, 1]
gini_norm_valid = gini_normalized(y_valid, pred_valid)

print('gini (normalised) for the validation set %s' % (round(gini_norm_valid , 4)))
preds = model.predict_proba(x_test)[:, 1]

os.chdir(sub_dir)

sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = preds

sub.to_csv(sample.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), gini_norm_valid),
                         index=False, float_format='%.4f'), index=False)