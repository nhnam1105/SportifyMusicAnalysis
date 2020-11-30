#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:09:13 2020

@author: namnguyen
"""
import statsmodels.api as sm
import numpy as np
import setting as s
import math

def bstrap(B, y, X):
    """
    Takes an array with dimensions (group, observation, variables).
    
    Outputs:
        * a bootstrapped mean for each group as lists
        * lower bounds of each bootstrapped mean as lists
        * upper bounds of each bootstrapped mean as lists
    """
    bs_params = []
    np.asarray(bs_params)
    for i in range(B):
        sample_index = np.random.choice(range(len(y)), len(y))
        X_samples = X.iloc[sample_index]
        y_samples = y.iloc[sample_index]
        ols = sm.OLS(y_samples, X_samples)
        ols_res = ols.fit()
        bs_params.append(ols_res.params)
    return bs_params

def compute_CI(bs_params):
    bs_means = np.mean(bs_params, axis = 0)
    bs_sd = np.std(bs_params, axis = 0)
    number_samples = len(bs_params)
    upper_bounds = []
    lower_bounds = []
    np.asarray(upper_bounds)
    np.asarray(lower_bounds)
    
    for i in range(len(bs_means)):
        lower_bound = bs_means[i] - (s.z_score * bs_sd[i])/math.sqrt(number_samples)
        upper_bound = bs_means[i] + (s.z_score * bs_sd[i])/math.sqrt(number_samples)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        
    return bs_means, lower_bounds, upper_bounds