#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:24:00 2018

@author: amc1354
"""

#clean some warning messages
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
#Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor



#store col names
names = ['idnum', 'age', 'workerclass', 'interestincome', 'traveltimetowork', 'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'educationalattain', 'sex', 'workarrivaltime', 'hoursworkperweek', 'ancestry', 'degreefield', 'industryworkedin', 'wages']


#import data
train_data = pd.read_csv('census_train.csv', names=names)
test_data = pd.read_csv('census_test.csv', names=names[:16])
#Set idnum as index to later keep prediction referring to the right idnum
train_data = train_data.set_index('idnum')
test_data = test_data.set_index('idnum')


#define features engineering in 2 functions. one we manually process features as described in the report. 2 we do one-hot encoding and standardize variables.
def featengineer_step1(df):
    #WE REMOVE non-workers defining them by hours per week worked. We will predict these with wage=0
    df = df.loc[df['hoursworkperweek']!='?']
    #WE ASSIGN 0 TO TRAVEL TIME = ? BECAUSE WE REMOVE NON-WORKERS SO THE REMAINING ONES WORK FROM HOME AND MAKES SENSE TO SET AN ORDINAL NUMBER AT 0 TO DIFFERENTIATE WITH OTHER TRAVELLING TIMES.
    df.loc[df['traveltimetowork']=='?', 'traveltimetowork'] = 0
    df.loc[df['vehicleoccupancy']!='1', 'vehicleoccupancy'] = 'other'
    df.loc[df['vehicleoccupancy']=='1', 'vehicleoccupancy'] = 'alonebycar'   
    df.loc[df['workarrivaltime']=='?', 'workarrivaltime'] = 0
    df.workarrivaltime = df['workarrivaltime'].astype(int)
    df.loc[(df['workarrivaltime']>93) & (df['workarrivaltime']<=225), 'workshift'] = 'dayarrival'
    df.loc[(df['workarrivaltime']<=93) | (df['workarrivaltime']>225), 'workshift'] = 'nightarrival'
    df.loc[df['workarrivaltime']==0, 'workshift'] = 'wfh'
    df.loc[df['workarrivaltime']==0, 'workarrivaltime'] = np.median(df.workarrivaltime[df['workarrivaltime']!=0])
    
    #all other variables have ? sorted  by now or have a meaning that we want to keep.
    #some cat var became numerical once removing the non-workers. check:
    #    df.info() #ancestry for example
    df.ancestry = df['ancestry'].astype(object)
    df.workshift = df['workshift'].astype(object)
    
    return(df)
    
def featengineer(X_train, X_test):
    
    X_train, X_test = featengineer_step1(X_train), featengineer_step1(X_test)
    numeric_enc = LabelEncoder()
    onehot_enc=OneHotEncoder(sparse=False)
    
    categorical_attrs = ['workerclass', 'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'sex', 'ancestry', 'degreefield', 'industryworkedin', 'workshift']
    
    X_train_temp = X_train.drop(categorical_attrs, axis=1)
    X_test_temp = X_test.drop(categorical_attrs, axis=1)
    
    for attr in categorical_attrs:
           # create a list of all possible categorical values (appearing in both train and test)
           data = X_train[[attr]].append(X_test[[attr]])
           # first convert to numerical, then to one-hot
           numeric_enc.fit(data)
           data_int_orig = numeric_enc.transform(X_train[[attr]])
           data_int = numeric_enc.transform(data)
           # Fit One Hot Encoding on train data
           onehot_enc.fit(data_int.reshape(-1, 1))
           temp = onehot_enc.transform(data_int_orig.reshape(-1, 1))
           # Changing the encoded features into a data frame with new column names
           temp = pd.DataFrame(temp, columns=[(attr+"_"+str(i)) for i in data[attr]
                .value_counts().index])
           # Set the index values same as the X_train data frame to concatenate correctly
           temp = temp.set_index(X_train.index.values)
           # adding the new One Hot Encoded varibales to the train data frame
           X_train_temp = pd.concat([X_train_temp, temp], axis=1)
           
           # Reoeat on test data
           data_int_orig = numeric_enc.transform(X_test[[attr]])
           data_int = numeric_enc.transform(data)
           onehot_enc.fit(data_int.reshape(-1, 1))
           temp = onehot_enc.transform(data_int_orig.reshape(-1, 1))
           temp = pd.DataFrame(temp, columns=[(attr+"_"+str(i)) for i in data[attr]
                .value_counts().index])
           temp = temp.set_index(X_test.index.values)
           X_test_temp = pd.concat([X_test_temp, temp], axis=1)
   
    # Standardizing the data set (need to save cols, and ids becasue scale functions gives back np array. we want to keep pandas DataFrame with same id's and col's as train and test data)
    id_train = X_train_temp.index
    col_train = X_train_temp.columns
    id_test = X_test_temp.index
    col_test = X_test_temp.columns
    
    X_train = pd.DataFrame(scale(X_train_temp), index=id_train, columns=col_train)
    X_test = pd.DataFrame(scale(X_test_temp), index=id_test, columns=col_test)
    
    return(X_train, X_test)
 
#split attributes and dependent variable    
X_train, y_train = train_data.loc[train_data['hoursworkperweek']!='?', names[1:16]], train_data.wages[train_data['hoursworkperweek'] != '?']
#run preprocessing on train and test data
X_train, X_test = featengineer(X_train, test_data)

#set seed for reproducibility of predictions
np.random.seed(45)
#use RF model with parameters decided with CV analyis as explained in the report
cvrf = RandomForestRegressor(n_estimators=200, max_depth=200)
cvrf.fit(X_train, y_train)

#build output file with the requested format
test_outputs_model = pd.DataFrame({'Id': test_data.loc[test_data['hoursworkperweek']!='?'].index.values,
                             'Wages': cvrf.predict(X_test)})
test_outputs_0 = pd.DataFrame({'Id': test_data.loc[test_data['hoursworkperweek']=='?'].index.values,
                             'Wages': np.zeros(test_data.shape[0]-test_outputs_model.shape[0])})
test_outputs = test_outputs_model.append(test_outputs_0)
test_outputs = test_outputs.sort_values(by=['Id'])
test_outputs.to_csv('test_outputs.csv', index=False)