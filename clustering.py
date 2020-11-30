#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:50:22 2020

@author: namnguyen
"""
from sklearn.mixture import GaussianMixture

def gmm_cluster(X, n_components):
    gmm = GaussianMixture(n_components).fit(X)
    pred_labels = gmm.predict(X)
    return pred_labels