# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 08:07:23 2017

@author: Evan
"""

def col_desc(df, col):
    
    unique_vals = df[col].nunique()
    quants = df[col].quantile([.25, .5, .75, .95])
    ttl_miss = df[col][(df[col]== -1)].sum() * -1
                
    var_res = {'name': col, 
               'ttl_obs': df[col].count(),
               'missing_obs': ttl_miss,
               'unique_vals': unique_vals,
               'perc_25': quants.iloc[0],
               'perc_50': quants.iloc[1],
               'perc_75': quants.iloc[2],
               'perc_95': quants.iloc[3]}
    
    return var_res
        

def var_desc(df, feat_suffixes = None):
    
    import pandas as pd
    
    var_desc_df = pd.DataFrame(columns = ['name', 'ttl_obs', 'missing_obs','unique_vals', 
                                   'perc_25', 'perc_50', 'perc_75', 'perc_95'])
    
    if feat_suffixes is not None:
    
        for suffix in feat_suffixes:
            
            cols = [i for e in suffix for i in df.columns  if e in i]
        
            for col in cols:
                
                var_res = col_desc(df, col)
                var_desc_df = var_desc_df.append(var_res, ignore_index = True)

    else:
        
        cols = df.columns
        
        for col in cols:
            
            var_res = col_desc(df, col)
            var_desc_df = var_desc_df.append(var_res, ignore_index = True)
            
    return var_desc_df.drop_duplicates()
        
        