# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:46:05 2017

@author: Evan
"""

import pandas as pd
import os

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Porto'
os.chdir(base_dir)

train = pd.read_csv('train.csv', low_memory = True)
test = pd.read_csv('test.csv', low_memory = True)

train_columns = list(train.columns.values)
test_columns = list(test.columns.values)

print(list(set(train_columns) - set(test_columns)))

y_train = train['target'].values
x_train = train.drop('target', axis = 1)

y_test= test['target'].values
x_test = test.drop('target', axis = 1)

binary_cols = [i for e in ['bin'] for i in list(train_columns) if e in i]
cat_cols = [i for e in ['cat'] for i in list(train_columns) if e in i]

