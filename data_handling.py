#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:48:47 2020

@author: namnguyen
"""
import pandas as pd
import setting as s

def read_data (filename: str):
    data = pd.read_csv(filename).iloc[1:s.number_observations]
    X = data[s.independent_variables]
    y = data[s.dependent_variables]
    return y, X