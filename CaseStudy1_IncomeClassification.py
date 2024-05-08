# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:16:08 2024

@author: rejoi
"""

import os # to set up os
import numpy as np # to perform numerical operations
import matplotlib as plt
import pandas as pd # to work with dataframes 
import seaborn as sns #to visualize data


# to partition the data (extracting the package modelselection from package sklearn)
from sklearn.model_selection import train_test_split 

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

data_income = pd.read_csv('income.csv') # to import data


print(data_income.head()) #displays first 5 rows of data
print(data_income.index) #displays no of rows
print(data_income.columns) #displays columns title
print(data_income.size) #displays total size of data (columns * rows)
print(data_income.shape) #displays no of columns and rows
print(data_income.memory_usage()) #displays memory size usage of each column 
print(data_income.ndim) #displays dimension size of data (columns and rows)


data_bkup=data_income.copy() # creating a copy of the data

"""
#Exploratory data analysis:

#1.Getting to know the data
#2.Data preprocessing (Missing values)
#3.Cross tables and data visualization
"""

print(data_bkup.info()) # displays datatype of each variable

#checking for missing values

data_bkup.isnull() # true shows missing data and false shows no missing data

print("Data values with null values is:\n", data_bkup.isnull().sum())
# therefore no missing values in any column
