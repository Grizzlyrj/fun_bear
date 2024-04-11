# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:16:03 2024

@author: rejoi
"""

import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\rejoi\\OneDrive\\Desktop\\Spyder Test Directory")

data_csv=pd.read_csv("sample1.csv",index_col=0,na_values=[4])

# sam=data_csv.copy(deep=False)

# deep_data=data_csv.copy(deep=True)

# data_csv.index
# data_csv.columns
# data_csv.size
# data_csv.shape
# data_csv.memory_usage()
# data_csv.ndim
# data_csv.head(5)
value=data_csv.at["Jul", ' "2006"']
print(value)

print(data_csv)

data_csv.info()

data_csv[' "2005"']=data_csv[' "2005"'].astype(float)

print(data_csv.columns)
