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

# =============================================================================
# Model Building
# =============================================================================

"""
We are going to build a Linear Regression and Random Forest model
on two sets of data.
1. Data obtained by omitting rows with any missing value
2. Data obtained by imputing the missing values 
"""

# =============================================================================
# OMITTING MISSING VALUES
# =============================================================================

cars_omit=cars_trial.dropna(axis=0) #axis=0 cause we want to drop any row that contains missing values.

'''9888 records with missing values dropped/removed'''

# Converting categorical variables to dummy variables

cars_omit=pd.get_dummies(data=cars_omit,drop_first=True)

# Convert True/False to 1/0 for all boolean columns

cars_omit = cars_omit.astype(float)

# =============================================================================
# IMPORTING NECESSARY LIBRARIES
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
{from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# =============================================================================
# MODEL BUILDING WITH OMITTED DATA
# =============================================================================

# Separating input and output features (x1=input, y1=output)

x1= cars_omit.drop(['price'], axis='columns', inplace=False) #inplace is false cause we dont want the changes 
                                                             #to happen in cars_omit
y1=cars_omit['price']


# Plotting the variable price 

'''Here we convert the price into its natural logarithmic value to obtain a bell shaped histogram. 
   The graph becomes more clearer to interpret then.'''

prices=pd.DataFrame({'1.Before log conversion':y1,'2.After log conversion':np.log(y1)})
prices.hist()

# Transforming price as a logarithmic value

y1=np.log(y1)

# Splitting data into test and train

x_train, x_test, y_train, y_test= train_test_split(x1, y1, test_size=0.3, random_state=3)

'''Here, test size contains 30% of the data while rest is used for training.
   random_state is 3 as the same third set of random sample of data while always be used for the train and 
    test when its been executed'''
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

'''Out: (23018, 300) (9866, 300) (23018,) (9866,)
    23018/32884=0.699 that is 70% for training
    9866/32884=0.300 that is 30% for testing
'''
# =============================================================================
# BASELINE MODEL FOR OMITTED DATA
# =============================================================================

"""
We are making a base model. The predicted value is basically replaced by using test data mean value
This is to set a benchmark and to compare with our regression model
"""
# finding the mean for test data value

base_pred=np.mean(y_test)

'''We found one base predicted value. Now we have to replecate the base predicted value for all 9866 rows'''

# Repeating same value till length of test data (9866)

base_pred=np.repeat(base_pred,len(y_test))

# finding the RMSE value (Root Mean Square Error)

'''RMSE is the square root of the mean square error. It computes the difference btw test value and predicted value,
    squares them and divides them by the number of samples. We get the mean square value then (MSE). Then we root
    the to get RMSE'''
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))

print(base_root_mean_square_error)

'''This is the benchmark for comparison with other modals. The objective in building future models is to get 
   an RSME lesser than this.'''

# =============================================================================
# LINEAR REGRESSION WITH OMITTED DATA
# =============================================================================

# Setting intercept as True

lgr=LinearRegression(fit_intercept=True)

'''fit_intercept=True: This parameter specifies whether to calculate the intercept for the model. 
   If set to True, the model will calculate the intercept, meaning it will fit the line such that it does not 
   necessarily pass through the origin. If set to False, the line will be forced to pass through the origin 
   (i.e., the intercept is set to 0).'''

# Model
model_lin=lgr.fit(x_train, y_train)

# Predicting model on test set
car_prediction_lin=lgr.predict(x_test)

# Computing MSE and RMSE
lin_mse=mean_squared_error(y_test, car_prediction_lin)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)
'''Out: 0.5455481266513825. Theresfore, there is almost a 50% drop from the base prediction error wise '''

# R squared value
'''The R-squared value, also known as the coefficient of determination, is a statistical measure used to 
evaluate the goodness of fit of a regression model. It indicates how well the independent variables explain 
the variability of the dependent variable in the model.  '''

r2_lin_test=model_lin.score(x_test,y_test)
r2_lin_train=model_lin.score(x_train,y_train)

print(r2_lin_test,r2_lin_train)

# Regression diagnostics- Residual plot analysis
'''Residual calculates the difference btw the test data and prediction'''

residual=y_test-car_prediction_lin
sns.regplot(x=car_prediction_lin, y=residual, scatter=True, fit_reg=False)
residual.describe()
'''Most of the residuals here are close to 0. Since residuals are errors, its much better to have most or atleast
   all to be 0 or close to it. Therefore, predicted and actual values are much closer'''
   

# =============================================================================
# RANDOM FOREST WITH OMITTED DATA
# =============================================================================

# Model parameters
