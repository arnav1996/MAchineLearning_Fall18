#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 09:18:24 2018

@authors: amc1385, ads798
"""




# libraries
import pandas as pd
import math
##################################




# custom functions
def mu(x):
	return sum(x)/len(x)
def sd(x):
    mean = mu(x)
    ss = (x - mean)**2
    var = ss.sum()/float(len(x)-1)
    return math.sqrt(var)
def calcProb(x, mu, sd):
	return(1 / (math.sqrt(2*math.pi) * sd)) * math.exp(-(math.pow(x-mu,2)/(2*sd**2)))
##################################
 
   
    
    
# Importing the dataset
dataset_train, dataset_test = pd.read_csv('spambasetrain.csv', header=None), pd.read_csv('spambasetest.csv', header=None)

# Split X and y in train and test
X = dataset_train.iloc[:, 0:9].values 
y = dataset_train.iloc[:, 9].values
Xt = dataset_test.iloc[:, 0:9].values 
yt = dataset_test.iloc[:, 9].values
##################################


# Here starts the session of the main calculations


# define the prior of class 'spam' (1) and class 'ham' (0)
prior_class_spam = sum(y)/len(y)
prior_class_ham = 1 - prior_class_spam 


# separate dataset by class
X_spam = X[y==1]
X_ham = X[y==0]


# then calculate mu and sd for the examples split by class.
mu_spam = {}
sd_spam = {}
for col in range(0, len(X_spam[0])):
	mu_spam[col], sd_spam[col] = mu(X_spam[:, col]), sd(X_spam[:, col])
    
mu_ham = {}
sd_ham = {}
for col in range(0, len(X_ham[0])):
	mu_ham[col], sd_ham[col] = mu(X_ham[:, col]), sd(X_ham[:, col])


Nt = len(yt)

# Predict: for each row in test, calculate probabilities and pick the class of the maximum probability calculated
y_pred = [0]*Nt
for i in range(0, Nt):
    prob_spam = prior_class_spam
    prob_ham = prior_class_ham
    for j in range(0, len(Xt[0])):
        prob_spam *= calcProb(Xt[i,j], mu_spam[j], sd_spam[j])
        prob_ham *= calcProb(Xt[i,j], mu_ham[j], sd_ham[j])
    if prob_ham <= prob_spam:
        y_pred[i] = 1
    else:
        y_pred[i] = 0


# Apply Zero-R algorithm
if sum(y==0) <= sum(y==1):
    pred_zr = 1
else:
    pred_zr = 0
y_pred_zr = [pred_zr]*Nt 
##################################




# save requested output in output text file
with open("progoutput.txt", "w") as text_file:
    text_file.write("* The estimated value of P(C) for C = 1 (Spam) is {0};".format(prior_class_spam))
    text_file.write("\n  the estimated value of P(C) for C = 0 (Ham) is {0};".format(prior_class_ham))
    text_file.write("\n* The estimates of the parameters mu and sigma for class C = 1 (Spam), for each variables X1, X2, ... X9 are, respectively mu: {0} and stdev: {1}.".format(mu_spam, sd_spam))
    text_file.write("\n  The estimates of the parameters mu and sigma for class C = 0 (Ham), for each variables X1, X2, ... X9 are, respectively mu: {0} and stdev: {1}.".format(mu_ham, sd_ham))
    text_file.write("\n* The predicted classes for all the test examples are, following the order of the test set, {}".format(y_pred))
    text_file.write("\n* Total number of test examples classified correctly is: {}.".format(sum(yt==y_pred)))
    text_file.write("\n* Total number of test examples classified incorrectly is: {}.".format(sum(yt!=y_pred)))
    text_file.write("\n* Hence, the percentage error on the test examples is: {}%.".format(100*sum(yt!=y_pred)/Nt))
text_file.close()
##################################




# print answers to put into proganswers.pdf
print("prior class 1 (spam) = ", prior_class_spam)
print("prior class 0 (not-spam) = ", prior_class_ham)
print("mean(X8|C=1(spam)) = ", mu_spam[7])
print("var(X8|C=1(spam)) = ", sd_spam[7]**2)
print("mean(X1|C=0(not-spam)) = ", mu_ham[0])
print("var(X1|C=0(not-spam)) = ", sd_ham[0]**2)
print("first five predicted labels: ", y_pred[0:5])
print("last five predicted labesl: ", y_pred[(Nt-5):Nt])
print("percentage error = {}%".format(100*sum(yt!=y_pred)/Nt))
print("accuracy using zero-r = {}%".format(100*sum(yt==y_pred_zr)/Nt))
############# EOF #####################