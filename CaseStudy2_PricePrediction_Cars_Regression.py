# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:36:06 2024

@author: rejoi
"""
'''
Objective: Develope an algorithm predicting price of cars based on various attributes associated with them.

Recorded data: Specification details, condition of car, seller details, Registration details,
               Web advertisement details, Make and model information, Price.

'''

# =============================================================================
# PREDICTING PRICE OF PRE-OWNED CARS 
# =============================================================================

import pandas as pd
import os
import numpy as np
import seaborn as sns

# =============================================================================
# Setting dimensions for plot 
# =============================================================================

sns.set(rc={'figure.figsize':(11.7,8.27)})
''' Here set function sets the size of the plots to be generated'''

# =============================================================================
# Reading CSV file
# =============================================================================
os.chdir(r'C:\Users\rejoi\OneDrive\Desktop\Github Folders\fun_bear')

print(os.getcwd())

cars_data=pd.read_csv('cars_sampled.csv')

# =============================================================================
# Creating copy
# =============================================================================

cars_copy=cars_data.copy() #Deep copy as any change in the copy does not reflect back in original

# =============================================================================
# Structure of the dataset
# =============================================================================
cars_copy.info()

# =============================================================================
# Summarizing data
# =============================================================================

cars_copy.describe()
'''Here , values are displayed in a scientific pov and with many decimal places. We will modify it 
to a simpler format'''
pd.set_option('display.float_format', lambda x: '%.3f' % x) 

'''Since all these are float values, we use display float format.
   lamda function is declared inside wth x denoting every value
   %.3f converts every x value to allocate only .3 decimal places'''
   
# Displaying max set of columns 

'''Currently, describe function only shows a few columns with the others labelled as '...'. So we change that'''

pd.set_option('display.max_columns',20)
cars_copy.describe()

# =============================================================================
# Dropping unwanted columns
# =============================================================================

cars_copy.columns

col=['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
cars_copy=cars_copy.drop(columns=col,axis=1)

# =============================================================================
# Removing duplicate records
# =============================================================================

cars_copy.drop_duplicates(keep='first', inplace=True) #Therefore, 470 duplicate records dropped

# =============================================================================
# Data cleaning
# =============================================================================

# No. of missing values in each column

cars_copy.isnull().sum()

# Variable yearOfRegistration

'Here, we take the year of registraion and count the frequency on each year'

yor_count=cars_copy['yearOfRegistration'].value_counts() # Displays in highest frequency order
yor_count=cars_copy['yearOfRegistration'].value_counts().sort_index() # Displays in year order

sum(cars_copy['yearOfRegistration'] > 2018) # 26 records
sum(cars_copy['yearOfRegistration'] < 1950) # 38 records

sns.regplot(data=cars_copy,x='yearOfRegistration', y='price', scatter=True,fit_reg=False)

'''So here, because of false data on years above and below 1950 to 2018, the graph smears out the useful info.
   Hence cleaning of the years is needed.
   
   Therefore the working range will be from 1950 to 2018'''
   
# Variable price
price_count=cars_copy['price'].value_counts().sort_index()
sns.distplot(cars_copy['price']) #Histogram skewed cause most data priced 0
cars_copy['price'].describe()
sns.boxplot(y=cars_copy['price']) #Box plot skewed cause most data priced 0
sum(cars_copy['price'] >150000) #34 records
sum(cars_copy['price']<100) #1748 records

'''So here, because of false data on price being 0, the graph smears out the useful info.
   Hence cleaning of price is needed.
   
   Therefore the working range will be from 100 to 150000'''

# Variable powerPS
cars_copy.columns
power_count=cars_copy['powerPS'].value_counts().sort_index()
sns.distplot(cars_copy['powerPS']) #Histogram skewed cause most data priced 0
cars_copy['powerPS'].describe()
sns.boxplot(y=cars_copy['powerPS']) #Box plot skewed cause most data priced 0
sns.regplot(data=cars_copy,x='powerPS',y='price',scatter=True,fit_reg=False)
sum(cars_copy['powerPS']>500)  #115 records
sum(cars_copy['powerPS']<10) #5565 records

'''So here, because of false data on price being 0, the graph smears out the useful info.
   Hence cleaning of powerPS is needed.
   
   Therefore the working range will be from 10 to 500'''

# =============================================================================
# Beginning the Data cleaning process
# =============================================================================   
