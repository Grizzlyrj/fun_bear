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

cars_trial=cars_copy[  (cars_copy['yearOfRegistration']<=2018)
                     & (cars_copy['yearOfRegistration']>=1950)
                     & (cars_copy['price']>=100)
                     & (cars_copy['price']<=150000)
                     & (cars_copy['powerPS']>=10)
                     & (cars_copy['powerPS']<=500)
                     ] #6759 records dropped

#Calculating the age of each car by yearOfRegistraion and monthOfRegistration
'''This process is added to simplify variable reduction'''

#Converting monthOfRegistration to decimal value
cars_trial.columns
cars_trial['monthOfRegistration']/=12

#Creating new variable 'Age' by adding yearOfRegistration and monthOfRegistration

cars_trial['Age']=(2018-cars_trial['yearOfRegistration']+cars_trial['monthOfRegistration'])
cars_trial['Age']=round(cars_trial['Age'],2)
cars_trial['Age'].describe()

# Dropping yearOfRegistration and monthOfRegistration

'''Since age is introduced, both the others can be dropped to avoid redundancy'''

cars_trial=cars_trial.drop(columns=['monthOfRegistration','yearOfRegistration'],axis=1)


# Visualizing parameters 

# Age

sns.distplot(cars_trial['Age']) #Histogram shows reliable info
sns.boxplot(y=cars_trial['Age']) #Boxplot shows reliable info

# Price

sns.distplot(cars_trial['price']) #Histogram shows reliable info
sns.boxplot(y=cars_trial['price']) #Boxplot shows reliable info

# PowerPS

sns.distplot(cars_trial['powerPS']) #Histogram shows reliable info
sns.boxplot(y=cars_trial['powerPS']) #Boxplot shows reliable info

# Visualizing parameters after narrowing working range
# Age vs price

sns.regplot(data=cars_trial,x='Age',y='price', scatter=True, fit_reg= False)

'''
Cars priced higher are newer
With increase in age, price decreases
However some cars are priced higher with increase in age'''

# powerPS vs price

sns.regplot(data=cars_trial, x='powerPS', y='price',scatter=True, fit_reg=False)

'''Through the graph, an increase in powerPS results in higher price'''



# Checking if other variables relate to price

cars_trial.columns
# Variable seller

cars_trial['seller'].value_counts()
pd.crosstab(cars_trial['seller'],columns='count', normalize=True)
sns.countplot(data=cars_trial,x='seller')

'''Fewer cars are in commercial. Therefore, variable seller is insignificant for relation'''

# Variable offerType

cars_trial['offerType'].value_counts()
pd.crosstab(cars_trial['offerType'],columns='count', normalize=True)
sns.countplot(data=cars_trial,x='offerType')

'''All cars are on offer. Therefore, variable offerType is insignificant for relation'''

# Variable abtest

cars_trial['abtest'].value_counts()
pd.crosstab(cars_trial['abtest'],columns='count', normalize=True)
sns.countplot(data=cars_trial,x='abtest')
'''Equally distributed. Therefore we check the boxplot with price relation'''
sns.boxplot(data=cars_trial,x='abtest',y='price')

'''The price seems more or less the same for both the abtest types. Therefore, this variable is also insignificant 
for relation'''

# Variable vehicleType

cars_trial['vehicleType'].value_counts()
pd.crosstab(cars_trial['vehicleType'],columns='count', normalize=True)
sns.countplot(data=cars_trial,x='vehicleType')
'''difference. Therefore, we check the boxplot with price relation'''
sns.boxplot(data=cars_trial,x='vehicleType',y='price')
sns.boxplot(data=cars_trial,x='price',y='vehicleType')
'''Since the price median is different for each type. We find that vehicleType influences the price'''

# Variable gearbox

cars_trial['gearbox'].value_counts()
pd.crosstab(cars_trial['gearbox'],columns='count', normalize=True)
sns.countplot(data=cars_trial,x='gearbox')
sns.boxplot(data=cars_trial,x='gearbox',y='price')

'''gearbox affects price'''

# Variable model

cars_trial['model'].value_counts()
pd.crosstab(cars_trial['model'],columns='count', normalize=True)
sns.countplot(data=cars_trial,y='model')
sns.boxplot(data=cars_trial,y='model',x='price')

'''model affect price.Cars are distributed over many models. Considered in modelling'''

# Variable kilometer

cars_trial['kilometer'].value_counts()
pd.crosstab(cars_trial['kilometer'],columns='count', normalize=True)
sns.countplot(data=cars_trial,y='kilometer')
sns.boxplot(data=cars_trial,y='kilometer',x='price')
'''boxplot unreadable'''
cars_trial['kilometer'].describe()
sns.displot(data=cars_trial,x='kilometer',bins=8, kde=False)
sns.regplot(data=cars_trial,x='kilometer',y='price', scatter=True,fit_reg=False)
'''Since different kilometer produce different prices, it is considered for modelling'''

# Variable fuelType

cars_trial['fuelType'].value_counts()
pd.crosstab(cars_trial['fuelType'],columns='count', normalize=True)
sns.countplot(data=cars_trial,x='fuelType')
sns.boxplot(data=cars_trial,y='fuelType',x='price')
'''fuelType affects price'''

# Variable brand

cars_trial['brand'].value_counts()
pd.crosstab(cars_trial['brand'],columns='count', normalize=True)
sns.countplot(data=cars_trial,y='brand')
sns.boxplot(data=cars_trial,y='brand',x='price')
'''brand affects price'''

# Variable notRepairedDamage

# yes- car is damaged but not rectified
# no- car was damaged but has been rectified

cars_trial['notRepairedDamage'].value_counts()
pd.crosstab(cars_trial['notRepairedDamage'],columns='count', normalize=True)
sns.countplot(data=cars_trial,y='notRepairedDamage')
sns.boxplot(data=cars_trial,x='notRepairedDamage',y='price')
'''As expected, the cars that require the damages to be repaired fall under lower price ranges.
    Therefore, notRepairedDamage affects price'''
    
# From this, we know that everything except abtest,seller and offerType, affects the price.

# =============================================================================
# Removing insignificant variables
# =============================================================================    

col=['seller','abtest','offerType']
cars_trial=cars_trial.drop(columns=col,axis=1)
cars_relvar=cars_trial.copy()

# =============================================================================
# Correlation between numerical variables
# =============================================================================

cars_select=cars_trial.select_dtypes(exclude=[object])
correlation=cars_select.corr()
round(correlation,3)
cars_select.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
'''here, .loc[:,'price'] takes only the first column, i.e, till price
   .abs() gives the absolute values
   [1:] skips the first row and displays the rest
   '''
# From the above data, only powerPS has more of an inclination to price changes, compared to the other two. But
# all 3 are still not heavily influenced by the price.


