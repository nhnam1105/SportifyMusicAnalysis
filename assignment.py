#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:44:38 2020

@author: namnguyen
"""

from data_handling import read_data
import setting as s
import pandas as pd
from sklearn.metrics import classification_report
import MLELinear as mlel
import plotting as p
import bootstrapping
import clustering
import evaluating

def main():
    
    #Read data and plit X and y
    y, X = read_data(s.data_file_path)

    #Implement a linear regressor based on Maximum Likelihood Estimation
    lm_res = mlel.fitLinearRegression(y, X)

    #show linear regressor summary
    print(lm_res.summary())
    
    #Estimating predicted labels
    y_hat = mlel.yhat(X, lm_res)
    
    #Plot y versus y predicted
    p.plot(y, y_hat)
    
    #compute L1
    l1 = mlel.compute_L1(y, y_hat)
    print('L1 error: ', l1[0])
    
    #Compute error between y and y_hat
    error = mlel.error_list(y, y_hat)
    
    #Plot y versus y predicted and error
    p.plot_error(error)
    
    #Bootstraping and we obtain params
    bs_params = bootstrapping.bstrap(s.number_replication, y, X)
    
    #get Means, lower and upper bounds
    means, lower_bounds, upper_bounds = bootstrapping.compute_CI(bs_params)
    
    print('Lower bounds: ', lower_bounds)
    print('Upper bounds:', upper_bounds)
    #Plot Confidence interval
    p.plotCI(np.asarray(bs_params), lower_bounds, upper_bounds)
    
    #Cluster method
    gmm_pred = clustering.gmm_cluster(X, s.n_components)
    #Report
    print(classification_report(y, gmm_pred, target_names=s.target_names))

    
if __name__ == "__main__":
    main()
