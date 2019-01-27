#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:39:44 2018

@author: amc
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dfply import *
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

names = ['idnum', 'age', 'workerclass', 'interestincome', 'traveltimetowork', 'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'educationalattain', 'sex', 'workarrivaltime', 'hoursworkperweek', 'ancestry', 'degreefield', 'industryworkedin', 'wages']

train_data = pd.read_csv('census_train.csv', names=names)

train_data.info()

train_data.wages[train_data['wages']>0].hist(bins=50)
np.log(train_data.wages[train_data['wages']>0]).hist(bins=50)
np.log(train_data.wages[train_data['wages']>100000]).hist(bins=50)
train_data.wages[train_data['wages']>100000].hist(bins=50)

scatter_matrix(train_data[['age','wages']])

train_data['age'].hist()
train_data["age"].value_counts()
np.quantile(train_data["age"], [0, 0.2, .5, .8, .9, 1])

#plt.show()


np.quantile(train_data['wages'], [0.2, .5, .8, .9, 1])
train_data[train_data.wages>90000].sum()['wages'] / train_data.wages.sum() #this apporach is wastage it sums all vars but I need only 1.
#Almost Pareto law - the top 20% combined earn 70% of the total income!. Highly skewed distn'. One experiment could be trying to model everything without knowing this. Then taking log transform (to help seeing more variability). Then taking 2 different models one for top20%, one for remaining. This can be achieved first by fitting a cluster method (knn or kmeans) to identify the top 20%, If this achieve high accuracy, we can apply the same to the training set.

#Explore data
train_data.isnull().sum()
train_data.info()
train_data["workerclass"].value_counts()
workerclasses = train_data['workerclass'].unique()

medians = np.array([])
for workerclass in workerclasses:
    medians = np.append(medians, np.median(train_data.wages[train_data['workerclass'] == workerclass]))
tmp = pd.DataFrame({'workclass': workerclasses, 'medianwage': medians})
tmp >> arrange(X.medianwage)
    
for workerclass in workerclasses:
    # Subset to the airline
    subset = train_data[train_data['workerclass'] == workerclass]
    
    # Draw the density plot
    sns.distplot(np.log(subset['wages']), hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = workerclass)
    
subset = train_data[train_data['workerclass'] == '6']
sns.distplot(subset['wages'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3},
                 label = '6')
# THERE ARE 3 GROUPS AROUND 50k, 300k, 650k

train_data.boxplot('wages', by='workerclass')


train_data['interestincome'].hist()

train_data["traveltimetowork"].value_counts()
subset = train_data[train_data['traveltimetowork'] != '?']
subset['traveltimetowork'].astype(int)
#subset.plot.scatter(x='traveltimetowork', y='wages', c='DarkBlue')
sns.boxplot(x='traveltimetowork', y='wages', data=subset)

train_data["vehicleoccupancy"].value_counts()
subset = (train_data >>
          select(X.vehicleoccupancy, X.meansoftransport, X.age) >>
          mask(X.vehicleoccupancy == '?'))
subset.meansoftransport.value_counts()
#What we learn here is that we can choose 2 ideas for vehicle occupancy: 1) we put all the ? at 1's with the euristic that all other means of transports are taken alone, and we preserve the ordinal nature of the attribute. 2) We transform the attribute in 2 categories such as - alone by car vs. other. Given the small number of examples in the other categories, I'd go for option 2). We can than try 1) in an experiment.

train_data["workarrivaltime"].value_counts()
#workarrivaltime is a tricky one because we have 111 working from home:
subset = (train_data >>
          select(X.workarrivaltime, X.traveltimetowork) >>
          mask(X.workarrivaltime == '?'))
subset.traveltimetowork.value_counts()
# what we can do is to convert to numeric and impute an average or median to the variable (1) or create categories such as arrives from 8 am to 10 am, intervals or pick times, or try to define night shift vs. day shift (2). We can actually do both, adding a new feature "shifttype". Option (3), if we want to further complicate things, we can use a model such as knn to impute the arrival times of the nearest neighbours, but it is time consuming so we can leave this as last resort in case we have time to run this experiment too.
    
train_data["meansoftransport"].value_counts()
train_data["marital"].value_counts()
train_data["schoolenrollment"].value_counts()
train_data["educationalattain"].value_counts()
train_data["sex"].value_counts()
train_data["workarrivaltime"].value_counts()
train_data["hoursworkperweek"].value_counts()
train_data["ancestry"].value_counts()
train_data["degreefield"].value_counts()
train_data["industryworkedin"].value_counts()


def featengineer_step1(df):
    #WE REMOVE non-workers defining them by hours per week worked. We predict these with wage=0
    df = df.loc[df['hoursworkperweek']!='?']
    #WE ASSIGN - TO TRAVEL TIME = ? BECAUSE WE REMOVED NON-WORKERS SO THE REMAINING ONES WORK FROM HOME AND MAKES SENSE TO SET AN ORDINAL NUMBER AT 0 TO DIFFERENTIATE WITH OTHER TRAVELLING TIMES.
    df.loc[df['traveltimetowork']=='?', 'traveltimetowork'] = 0
    
    df.loc[df['vehicleoccupancy']!='1', 'vehicleoccupancy'] = 'other'
    df.loc[df['vehicleoccupancy']=='1', 'vehicleoccupancy'] = 'alonebycar'
    #From meansoftransport, we can keep it as it is. One possible way to treat it is to include in a category 'other' the groups with lower numbers of examples, e.g., 6,7,9,12
    
    df.loc[df['workarrivaltime']=='?', 'workarrivaltime'] = 0
    df.workarrivaltime = df['workarrivaltime'].astype(int)
    df.loc[(df['workarrivaltime']>93) & (df['workarrivaltime']<=225), 'workshift'] = 'dayarrival'
    df.loc[(df['workarrivaltime']<=93) | (df['workarrivaltime']>225), 'workshift'] = 'nightarrival'
    df.loc[df['workarrivaltime']==0, 'workshift'] = 'wfh'

    df.loc[df['workarrivaltime']==0, 'workarrivaltime'] = np.median(df.workarrivaltime[df['workarrivaltime']!=0])
    
    #all other variables have ? sorted or have a meaning that we want to keep.
    #some cat var became numerical once removing the non-workers. check:
#    df.info()
    #ancestry for example
    df.ancestry = df['ancestry'].astype(object)
    df.workshift = df['workshift'].astype(object)
    #finally, we don't need idnum
    df = (df >> drop(X.idnum))
    
    return(df)
    
def featengineer(X_train, X_test):
    
    X_train, X_test = featengineer_step1(X_train), featengineer_step1(X_test)
    numeric_enc = LabelEncoder()
    onehot_enc=OneHotEncoder(sparse=False)
    
    categorical_attrs = ['workerclass', 'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'sex', 'ancestry', 'degreefield', 'industryworkedin', 'workshift']
    
    X_train_temp = X_train.drop(categorical_attrs, axis=1)
    X_test_temp = X_test.drop(categorical_attrs, axis=1)
    
    for attr in categorical_attrs:
           # creating an exhaustive list of all possible categorical values
           data = X_train[[attr]].append(X_test[[attr]])
           numeric_enc.fit(data)
           data_int_orig = numeric_enc.transform(X_train[[attr]])
           data_int = numeric_enc.transform(data)
           # Fitting One Hot Encoding on train data
           onehot_enc.fit(data_int.reshape(-1, 1))
           temp = onehot_enc.transform(data_int_orig.reshape(-1, 1))
           # Changing the encoded features into a data frame with new column names
           temp = pd.DataFrame(temp, columns=[(attr+"_"+str(i)) for i in data[attr]
                .value_counts().index])
           # In side by side concatenation index values should be same
           # Setting the index values similar to the X_train data frame
           temp = temp.set_index(X_train.index.values)
           # adding the new One Hot Encoded varibales to the train data frame
           X_train_temp = pd.concat([X_train_temp, temp], axis=1)
           
           # fitting One Hot Encoding on test data
           data_int_orig = numeric_enc.transform(X_test[[attr]])
           data_int = numeric_enc.transform(data)
           # Fitting One Hot Encoding on train data
           onehot_enc.fit(data_int.reshape(-1, 1))
           temp = onehot_enc.transform(data_int_orig.reshape(-1, 1))
           # changing it into data frame and adding column names
           temp = pd.DataFrame(temp, columns=[(attr+"_"+str(i)) for i in data[attr]
                .value_counts().index])
           # Setting the index for proper concatenation
           temp = temp.set_index(X_test.index.values)
           # adding the new One Hot Encoded varibales to test data frame
           X_test_temp = pd.concat([X_test_temp, temp], axis=1)
   
    # Standardizing the data set (need to save cols, and ids becasue scale functions gives back np array. we want to keep pandas DataFrame with same id's and col's as train and test data)
    id_train = X_train_temp.index
    col_train = X_train_temp.columns
    id_test = X_test_temp.index
    col_test = X_test_temp.columns
    
    X_train = pd.DataFrame(scale(X_train_temp), index=id_train, columns=col_train)
    X_test = pd.DataFrame(scale(X_test_temp), index=id_test, columns=col_test)
    
    return(X_train, X_test)
 
    
test_data = pd.read_csv('census_test.csv', names=names[:16])
X_train, y_train = (train_data >> drop(X.wages)), train_data.wages[train_data['hoursworkperweek'] != '?']
X_train, X_test = featengineer(X_train, test_data)





from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=45)

###########Create benchmark####################3
from scipy import stats  

# create some normal random noisy data
ser = np.log(y_train[y_train!=0])
# plot normed histogram
plt.hist(ser, normed=True, bins=50)
# find minimum and maximum of xticks, so we know
# where we should compute theoretical distribution
xt = plt.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(ser))
# normal distribution
m, s = stats.norm.fit(ser) # get mean and standard deviation  
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
plt.plot(lnspc, pdf_g, label="Norm") # plot it
plt.show()

####### let's start playing with models ########       
   

np.random.seed(45)
y_benchmark = np.exp(np.random.normal(m, s, len(y_test)))

#Benchmark
mean_squared_error(y_test, y_benchmark)**.5/10**3
plt.scatter(y_test, y_benchmark,  color='black')


regr_rf = RandomForestRegressor(n_estimators=30, max_depth=100,
                                random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_rf = regr_rf.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_rf)**.5/10**3

# Plot the results
plt.scatter(y_test, y_rf,  color='black')



regr_ln = LinearRegression()
regr_ln.fit(X_train, y_train)

# Predict on new data
y_ln = regr_ln.predict(X_test)
mean_squared_error(y_test, y_ln)**.5/10**3
plt.scatter(y_test, y_ln,  color='blue')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge

def RMSE(y, yhat): return -mean_squared_error(y, yhat)**.5/10**3 # "-" beacause GridSearchCV maximize
score = make_scorer(RMSE)

#Choose the type of classifier. 
regr_ln_ridge_cv = Ridge()
# Choose some parameter combinations to try
parameters = {'alpha': [100, 1000, 10000]}
# Run the grid search
grid_obj = GridSearchCV(regr_ln_ridge_cv, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj = grid_obj.fit(X_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
regr_ln_ridge = grid_obj.best_estimator_
regr_ln_ridge.fit(X_train, y_train)
y_ln_ridge = regr_ln_ridge.predict(X_test)
print(-RMSE(y_test, y_ln_ridge))
plt.scatter(y_test, y_ln_ridge,  color='blue')



from sklearn.linear_model import Lasso
regr_ln_lasso_cv = Lasso(max_iter=5000)
parameters = {'alpha': [100, 1000, 10000, 20000]}
grid_obj = GridSearchCV(regr_ln_lasso_cv, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj = grid_obj.fit(X_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
regr_ln_lasso = grid_obj.best_estimator_
regr_ln_lasso.fit(X_train, y_train)
y_ln_lasso = regr_ln_lasso.predict(X_test)
print(-RMSE(y_test, y_ln_lasso))
plt.scatter(y_test, y_ln_lasso,  color='blue')


np.random.seed(45)
cvrf = RandomForestRegressor()
parameters = {'n_estimators': [100, 200, 300, 410],
              'max_depth': [50, 100, 200]
             }
grid_obj = GridSearchCV(cvrf, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj = grid_obj.fit(X_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
cvrf = grid_obj.best_estimator_
cvrf.fit(X_train, y_train)
y_clf = cvrf.predict(X_test)
print(-RMSE(y_test, y_clf))
plt.scatter(y_test, y_clf,  color='green')


from sklearn.neural_network import MLPRegressor
#learning_rate_init
##alpha
#hidden_layer_sizes
#n_iter_no_change

cvnnet = MLPRegressor()
parameters = {'learning_rate_init': [1, .01, .001],
              'hidden_layer_sizes': [50, 100],
              'max_iter': [10000]
             }
grid_obj = GridSearchCV(cvnnet, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj.fit(X_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
reg_nnet = grid_obj.best_estimator_
reg_nnet.fit(X_train, y_train)
y_nnet = reg_nnet.predict(X_test)
print(-RMSE(y_test, y_nnet))
plt.scatter(y_test, y_nnet,  color='purple')



############ PCA ################
X_train, y_train = (train_data >> drop(X.wages)), train_data.wages[train_data['hoursworkperweek'] != '?']
X_train, X_test = featengineer(X_train, test_data)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
pca.explained_variance_
sum(pca.explained_variance_ratio_[:300])
sum(pca.explained_variance_ratio_[:200])
sum(pca.explained_variance_ratio_[:150])

pca = PCA(n_components=200)
Z_train = pca.fit_transform(X_train)

Z_train, Z_test, y_train, y_test = train_test_split(Z_train, y_train, test_size=0.2, random_state=45)


regr_ln = LinearRegression()
regr_ln.fit(Z_train, y_train)
# Predict on new data
y_ln = regr_ln.predict(Z_test)
print(-RMSE(y_test, y_ln))
plt.scatter(y_test, y_ln,  color='blue')


regr_ln_ridge_cv = Ridge()
parameters = {'alpha': [100, 1000, 10000]}
grid_obj = GridSearchCV(regr_ln_ridge_cv, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj = grid_obj.fit(Z_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
regr_ln_ridge = grid_obj.best_estimator_
regr_ln_ridge.fit(Z_train, y_train)
y_ln_ridge = regr_ln_ridge.predict(Z_test)
print(-RMSE(y_test, y_ln_ridge))
plt.scatter(y_test, y_ln_ridge,  color='blue')



regr_ln_lasso_cv = Lasso(max_iter=5000)
parameters = {'alpha': [100, 1000, 10000, 20000]}
grid_obj = GridSearchCV(regr_ln_lasso_cv, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj = grid_obj.fit(Z_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
regr_ln_lasso = grid_obj.best_estimator_
regr_ln_lasso.fit(Z_train, y_train)
y_ln_lasso = regr_ln_lasso.predict(Z_test)
print(-RMSE(y_test, y_ln_lasso))
plt.scatter(y_test, y_ln_lasso,  color='blue')



cvrf = RandomForestRegressor()
parameters = {'n_estimators': [10, 50, 100, 150],
              'max_depth': [10, 50, 100]
             }
grid_obj = GridSearchCV(cvrf, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj = grid_obj.fit(Z_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
cvrf = grid_obj.best_estimator_
cvrf.fit(Z_train, y_train)
y_clf = cvrf.predict(Z_test)
print(-RMSE(y_test, y_clf))
plt.scatter(y_test, y_clf,  color='green')



cvnnet = MLPRegressor()
parameters = {'learning_rate_init': [1, .01, .001],
              'hidden_layer_sizes': [50, 100],
              'max_iter': [10000]
             }
grid_obj = GridSearchCV(cvnnet, parameters, cv=5, scoring=score, return_train_score=True)
grid_obj.fit(Z_train, y_train)
grid_obj.cv_results_
grid_obj.best_score_
grid_obj.best_estimator_
reg_nnet = grid_obj.best_estimator_
reg_nnet.fit(Z_train, y_train)
y_nnet = reg_nnet.predict(Z_test)
print(-RMSE(y_test, y_nnet))
plt.scatter(y_test, y_nnet,  color='purple')




