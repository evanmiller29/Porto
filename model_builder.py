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