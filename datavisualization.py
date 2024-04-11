# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:27:21 2024

@author: rejoi
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Load data from CSV file
data_csv = pd.read_csv("diamond.csv")

# # Print first few rows of the DataFrame to verify data loading
# print(data_csv.head())

# # Plotting
# plt.scatter(data_csv['2006'], data_csv['Average'], c='red')
# plt.title("Scatterplot")
# plt.xlabel("2006")
# plt.ylabel("Average")
# plt.show()

# num_data=data_csv.select_dtypes(exclude=[object])
# print(num_data.corr())

sns.boxplot(x="price",y='cut',data=data_csv)

data2=pd.read_csv("churn.csv")


data2.isna().sum()

data3=pd.read_csv("flavors_of_cocoa.csv")

data3.isnull().sum()

data3.value_counts("Company Location")

data3.describe()
