#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:50:41 2020

@author: namnguyen
"""

import matplotlib.pyplot as plt
import numpy as np

def plot(y, y_hat):
    plt.figure(1)
    ax = plt.axes()
    ax.plot(y, 'o', color='black')
    ax.plot(y_hat, 'o', color='yellow')

def plot_error(error):
    plt.figure(1)
    ax = plt.axes()
    ax.plot(error, '-', color='red')
    
def plotCI(res_params, up, low):
    plt.figure(2)
    height, bins, patches = plt.hist(res_params, alpha=0.3)
    plt.fill_betweenx([0, max(height)], low, up, color='g', alpha=0.1)
    
    