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

#Getting summary of numerical columns

print(data_bkup.describe())

summary_numdata=data_bkup.describe() 

#Getting summary of categorical columns

summary_catdata=data_bkup.describe(include="O") #here O represents object data type


print(summary_catdata)

# Frequency of each category

data_bkup['JobType'].value_counts() # there is ? in jobtype
data_bkup['occupation'].value_counts() # there is ? in occupation
data_bkup['nativecountry'].value_counts()
data_bkup['age'].value_counts()
data_bkup['capitalgain'].value_counts() #capital gain is the profit from 
                                        #investments gained by selling-purchase price  

data_bkup['capitalloss'].value_counts()#capital loss is the loss from 
                                        #investments calculated by selling-purchase price 
                                        
# Checking for unique values

print(np.unique(data_bkup['JobType']))
print(np.unique(data_bkup['occupation']))
#**** There exists ' ?' instesd of nan

"""
Go back and read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
"""

data_bkup = pd.read_csv('income.csv',na_values=[" ?"]) 

# =============================================================================
# Data pre-processing
# =============================================================================
data_bkup.isnull().sum()

"""
# Here we know that there are 1809 missing values from Jobtype and 1816 missing from occupation
We need to know how many rows are missing both column values. So we create a subset named 'missing'.
"""

missing=data_bkup[data_bkup.isnull().any(axis=1)] # axis=1 => to consider at least one column value is missing in a row

""" Points to note:
1. Missing values in Jobtype    = 1809
2. Missing values in Occupation = 1816 
3. There are 1809 rows where two specific 
   columns i.e. occupation & JobType have missing values
4. (1816-1809) = 7 => You still have occupation unfilled for 
   these 7 rows. Because, jobtype is Never worked
"""

data_com=data_bkup.dropna(axis=0) #Created new df that dropped all rows with nan and contains only complete values.

# Relationship between independent variables i.e numerical variables

correlation=data_com.corr()

print(data_com.dtypes)

numeric_data_com=data_com.select_dtypes(include=['float','int'])

correlation=numeric_data_com.corr()
"""
Upon checking correlation df, none of the combinations are closer to neither 1 nor -1. 
This shows that there is no strong correlation btw variables.

"""
# =============================================================================
# Cross tables & Data Visualization
# =============================================================================
# Extracting the column names

data_com.columns


# =============================================================================
# Gender proportion table:
# =============================================================================

gender=pd.crosstab(data_com["gender"],columns='count',normalize=True) # normalize helps in getting percent values
print(gender)

# =============================================================================
#  Gender vs Salary Status:
# =============================================================================

gender_salstat=pd.crosstab(data_com["gender"], columns=data_com["SalStat"], normalize=True, margins=True)
print(gender_salstat)

# =============================================================================
# Frequency distribution of 'Salary status' 
# =============================================================================
SalStat = sns.countplot(x='SalStat', data=data_com)

"""  75 % of people's salary status is <=50,000 
     & 25% of people's salary status is > 50,000
"""
##############  Histogram of Age  #############################

sns.distplot(data_com['age'],kde=False,bins=10)

# People with age 20-45 age are high in frequency

############# Box Plot - Age vs Salary status #################

sns.boxplot(data=data_com,x='SalStat', y='age') # creates boxplot of data
data_com.groupby('SalStat')['age'].median() # shows median age of both groups

## people with 35-50 age are more likely to earn > 50000 USD p.a
## people with 25-35 age are more likely to earn <= 50000 USD p.a

##############  Grouped barplot of Jobtype wrt SalStat  #############################
JobType= sns.countplot(y=data_com['JobType'],hue = 'SalStat', data=data_com)

job_salstat =pd.crosstab(index = data_com["JobType"],columns = data_com['SalStat'],margins=True, normalize='index')  
round(job_salstat*100,1) # here 1 represents to round the data to atleast 1 decimal point

# margins shows the total value at the end of each row and column
# normalize converts it into percent values according to each row (i.e index)

##############  Grouped barplot of Edtype wrt SalStat  #############################

Education= sns.countplot(y=data_com[ 'EdType'], hue= 'SalStat', data=data_com)
edtype_salstat=pd.crosstab(index=data_com['EdType'],columns=data_com['SalStat'], margins=True,normalize='index')
round(edtype_salstat*100,1)

# margins shows the total value at the end of each row and column
# normalize converts it into percent values according to each row (i.e index)

##############  Grouped barplot of Occupation wrt SalStat  #############################

occu= sns.countplot(y='occupation', hue='SalStat',data=data_com)
occu_salstat=pd.crosstab(index=data_com['occupation'],columns=data_com['SalStat'],margins=True,normalize='index')
round(occu_salstat*100,1)
data_com.columns

##############  Histogram of Capital gain  #############################

cp=sns.distplot(data_com['capitalgain'],bins=10,kde=False)

##############  Histogram of Capital loss #############################

cl=sns.distplot(data_com['capitalloss'],bins=10,kde=False)




