# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:16:08 2024

@author: rejoi
"""

import os
import numpy as np
import matplotlib as plt
import pandas as pd

data_csv=pd.read_csv("income.csv")

print(data_csv.head()) #displays first 5 rows of data
print(data_csv.index) #displays no of rows
print(data_csv.columns) #displays columns title
print(data_csv.size) #displays total size of data (columns * rows)
print(data_csv.shape) #displays no of columns and rows
print(data_csv.memory_usage()) #displays memory size usage of each column 
print(data_csv.ndim) #displays dimension size of data (columns and rows)


