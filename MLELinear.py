#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:29:10 2020

@author: namnguyen
"""

from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
import statsmodels.api as sm

class MLELinearRegression(GenericLikelihoodModel):
    
    def loglike(self, params):
        scale = params[-1]
        weights = params[:-1]
        y_hat = np.sum(weights * self.exog[:, :-1], axis=-1)
        return np.sum(stats.norm.logpdf(self.endog, loc=y_hat, scale=scale))

#Fit MLE Linear Regression
        # dp_var dataframe  dataframe of dependent variables
        # idp_var dataframe  dataframe of independent variables
def fitLinearRegression (dp_var, idp_var):
    lm = MLELinearRegression(dp_var, idp_var)
    return lm.fit()

#Estimate y hat based on dependent variable and limear model response
def yhat (X, lm_res):
    #lm_params = np.asmatrix((lm_res.params))
    X = np.asmatrix(X)
    y_pred = np.dot(X,lm_res.params).reshape(-1, 1)
    return y_pred

def compute_L1(y, y_hat):
    l1 = np.sum(np.abs(y-y_hat)) 
    return l1

def fitOLS(y, X):
    X['const'] = np.ones(len(X))
    ols = sm.OLS(y, X)
    ols_res = ols.fit()
    return ols_res

def error_list(y, y_hat):
    return np.abs(y - y_hat)
    