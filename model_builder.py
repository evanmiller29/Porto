import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from numba import jit
import os
from imblearn.over_sampling import SMOTE
from imblearn import pipeline as pl

# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
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

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Porto'
sub_dir = 'F:/Nerdy Stuff/Kaggle submissions/Porto'
script_dir = 'C:/Users/Evan/Documents/GitHub/Porto'

os.chdir(base_dir)

runDesc = {'MAX_ROUNDS': 650,
           'OPTIMIZE_ROUNDS': False,
           'LEARNING_RATE': 0.05,
           'K': 5,
           'CROSS_VAL': True}

train_df = pd.read_csv('train.csv', low_memory = True)
test_df = pd.read_csv('test.csv', low_memory = True)
sample = pd.read_csv('sample_submission.csv', low_memory = True)

# Process data
id_test = test_df['id'].values
id_train = train_df['id'].values

train_df = train_df.fillna(999)
test_df = test_df.fillna(999)

col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
cat_cols = [i for e in ['cat'] for i in train_df.columns  if e in i]
cat_cols_idx = [i for i, c in enumerate(train_df.columns) if 'cat' in c]

train_df = train_df.drop(col_to_drop, axis=1)  
test_df = test_df.drop(col_to_drop, axis=1)  

for c in train_df.select_dtypes(include=['float64']).columns:
    train_df[c]=train_df[c].astype(np.float32)
    test_df[c]=test_df[c].astype(np.float32)

for c in train_df.select_dtypes(include=['int64']).columns[2:]:
    train_df[c]=train_df[c].astype(np.int8)
    test_df[c]=test_df[c].astype(np.int8)
    
y = train_df['target']
X = train_df.drop(['target', 'id'], axis=1)
y_valid_pred = 0*y
X_test = test_df.drop(['id'], axis=1)
y_test_pred = 0

# Set up folds

gini_res = []

params = {'learning_rate' : runDesc['LEARNING_RATE'], 
            'depth' : 6, 
            'l2_leaf_reg' : 8, 
            'border_count' : 10,
            'ctr_border_count' : 50,
            'iterations' : runDesc['MAX_ROUNDS'],
            'thread_count' : 6,
            'loss_function' : 'Logloss'
        
        }

model = CatBoostClassifier(**params)

if runDesc['CROSS_VAL']:

    kf = KFold(n_splits = runDesc['K'], random_state = 1, shuffle = True)

    for i, (train_index, test_index) in enumerate(kf.split(train_df)):
        
        # Create data for this fold
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
        
        X_train, y_train = SMOTE(random_state=42).fit_sample(X_train, y_train)
        
        print( "\nFold ", i)
        
        # Run model for this fold
        if runDesc['OPTIMIZE_ROUNDS']:
            fit_model = model.fit( X_train, y_train, cat_features=cat_cols_idx,
                                   eval_set=[X_valid, y_valid],
                                   use_best_model=True
                                 )
            print( "  N trees = ", model.tree_count_ )
        else:
            fit_model = model.fit( X_train, y_train )
            
        # Generate validation predictions for this fold
        pred = fit_model.predict_proba(X_valid)[:,1]
        print( "  Gini = ", eval_gini(y_valid, pred) )
        y_valid_pred.iloc[test_index] = pred
        
        # Accumulate test set predictions
        y_test_pred += fit_model.predict_proba(X_test)[:,1]
        gini_res.append(eval_gini(y_valid, pred))
        
    y_test_pred /= runDesc['K']  # Average test set predictions
    
    print( "\nGini for full training set:" )
    print(eval_gini(y, y_valid_pred))
    
    avg_gini = str(round(np.mean(gini_res), 4))[0:10]

if not runDesc['CROSS_VAL']:
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, 
                                                        random_state=42)
    
    if runDesc['OPTIMIZE_ROUNDS']:
            fit_model = model.fit( X_train, y_train, cat_features=cat_cols_idx,
                                   eval_set=[X_valid, y_valid],
                                   use_best_model=True
                                 )
            print( "  N trees = ", model.tree_count_ )
    else:
        
        X_train, y_train = SMOTE(random_state=42).fit_sample(X_train, y_train)
        fit_model = model.fit( X_train, y_train )
            
        # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
            
    print( "\nGini for full training set:" )
    print(eval_gini(y, pred))
    
os.chdir(sub_dir)
# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred

sub.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), avg_gini),
                         index=False, float_format='%.6f')