#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:01:15 2020

@author: namnguyen
"""

from sklearn.metrics import classification_report
import numpy as np
from scipy.stats import mode

def classification_measure(true_values, pred_values):
    pred_values = collapse_components(true_values, pred_values)

    print(classification_report(true_values, pred_values))
    
def collapse_components(true_values, pred_values):
    new_pred = np.zeros_like(pred_values)
    for true_val in np.unique(true_values):
        # look for the most similar component in pred_values
        val, _ = mode(pred_values[true_values == true_val])
        new_pred[pred_values == val] = true_val 
    return new_pred