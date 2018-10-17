#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:08:59 2018

@author: amc1385, ads798
"""
 



# libraries
import numpy
import operator
import math
import pandas
import itertools
import statistics
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
##################################




# custom functions
def preprocess(review):
            # Stem and remove stopwords
            review = re.sub('[^a-zA-Z]', ' ', review)
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            return ' '.join(review)

def customknn(k, traindata, testdata, distance):
       
    y_pred = []
       
    if distance == '1/count':
        
        for testreview in testdata:
            testreview = testreview[1:]
            freq = {}
            n=-1  
            for trainreview in traindata:
                trainreview = trainreview[1:]
                count=0
                for word in trainreview:
                    if word in testreview:
                        count+=1
                n+=1
                freq.update({ n:(math.inf if count==0 else 1/count) })
                
            sorted_freq = sorted(freq.items(), key=operator.itemgetter(1))
        
            #note that here the decision is for inf to make the prediction a bit random. We may later adjust this to remove any inf distance and pick the only ones left if any, and if none predict 1.
            if sorted_freq[:k][k-1][1]!=sorted_freq[k][1] or sorted_freq[:k][k-1][1]==sorted_freq[k][1]==math.inf:
                selected = sorted_freq[:k]
            else:
                tmp = [j for i, j in sorted_freq ]
                indeces = numpy.where(numpy.roll(tmp,1)!=tmp)[0]
                selected = sorted_freq[:indeces[indeces>=k][0]]
            
            selected_traindata = list(traindata[z] for z in [i for i, j in selected])
            classes_for_pred = []
            for trainreview in selected_traindata:
                classes_for_pred.append(int(trainreview[0]))
            
            y_pred.append(1 if sum(classes_for_pred)/len(classes_for_pred)>=.5 else 0)
    
    
    
    elif distance == 'cosine':
        # We need a different data structure for this new distance metric.
        dataset = pandas.DataFrame(data = {'review': [' '.join(traindata[0][1:])], 'label': [int(traindata[0][0])]})
        for item in traindata[1:]:
            dataset = dataset.append(pandas.DataFrame([[' '.join(item[1:]), int(item[0])]], columns=['review','label']), ignore_index=True)

        testdataset = pandas.DataFrame(data = {'review': [' '.join(testdata[0][1:])], 'label': [int(testdata[0][0])]})
        for item in testdata[1:]:
            testdataset = testdataset.append(pandas.DataFrame([[' '.join(item[1:]), int(item[0])]], columns=['review','label']), ignore_index=True)

        trainingcorpus = []
        for i in range(0, len(dataset)):
            review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
            #review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            trainingcorpus.append(review)      
        
            # Creating the Bag of Words model with TFIDF and calc cosine_similarity
        vectorizer = CountVectorizer(decode_error="replace")
        vec_train = vectorizer.fit_transform(trainingcorpus) #this is needed to get the attribute vocabulary_
        training_vocabulary = vectorizer.vocabulary_
        transformer = TfidfTransformer()
        trainingvoc_vectorizer = CountVectorizer(decode_error="replace", vocabulary=training_vocabulary)
        tfidf_trainingcorpus = TfidfVectorizer().fit_transform(trainingcorpus)
        
        for review in testdataset['review']:
            tfidf_testexample = transformer.fit_transform(trainingvoc_vectorizer.fit_transform(numpy.array([preprocess(review)]))) 
            cosine_similarities = cosine_similarity(tfidf_testexample, tfidf_trainingcorpus)
            related_docs_indices = (-cosine_similarities[0]).argsort()
            sorted_freq = cosine_similarities[0][related_docs_indices]
            
            #note for this distance the problem we had befor with inf, we have now with 0. Again we decide
            #to make the prediction a bit random. This could be adjusted to remove any 0 distance and
            #pick the only ones left if any, and if none predict 1.
            if sorted_freq[k-1]!=sorted_freq[k] or sorted_freq[k-1]==sorted_freq[k]==0:
                selected = related_docs_indices[:k]
            else:
                indeces = numpy.where(numpy.roll(sorted_freq,1)!=sorted_freq)
                selected = related_docs_indices[:indeces[0][indeces[0]>=k][0]]
            
            classes_for_pred = list(dataset.iloc[selected]['label'])
            
            y_pred.append(1 if sum(classes_for_pred)/len(classes_for_pred)>=.5 else 0)
      
    return y_pred
    
    

##################################




# Upload data
with open("reviewstrain.txt") as f:
    traindata=[line.split() for line in f]

with open("reviewstest.txt") as f:
    testdata=[line.split() for line in f]
##################################

y_test = []
for review in testdata:
    y_test.append(int(review[0]))



# print answers to put into proganswers.pdf
    
####### (a) ###########

k=1

y_pred = customknn(k, traindata, testdata, distance = '1/count')
 
#(a)i.
print("(a)i. predicted label for: ", " ".join(testdata[17][1:]))
print("\n prediction class for k={}: ".format(k), y_pred[17])

#(a)ii.
y_test = pandas.Series(y_test)
y_pred = pandas.Series(y_pred)
cm = pandas.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("\n (a)ii. Confusion matrix for k={}:\n\n".format(k), cm)
print("\n TN = ", cm[0][0])
print("FP = ", cm[1][0])
print("FN = ", cm[0][1])
print("TP = ", cm[1][1])

#(a)iii.
print("\n (a)iii. Accuracy for k={}: ".format(k), (cm[0][0]+cm[1][1])/cm['All']['All'])
print("TPR (Sensitivity) for k={}: ".format(k), cm[1][1]/cm[1]['All'])
print("FPR (false alarm ratio) for k={}: ".format(k), cm[1][0]/cm['All'][0])

k=5

y_pred = customknn(k, traindata, testdata, distance = '1/count')
#(a)iv.
print("(a)iv. predicted label for: ", " ".join(testdata[17][1:]))
print("\n prediction class for k={}: ".format(k), y_pred[17])

#(a)v.
y_test = pandas.Series(y_test)
y_pred = pandas.Series(y_pred)
cm = pandas.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("\n (a)v. Confusion matrix for k={}:\n\n".format(k), cm)
print("\n TN = ", cm[0][0])
print("FP = ", cm[1][0])
print("FN = ", cm[0][1])
print("TP = ", cm[1][1])

#(a)vi.
print("\n (a)vi. Accuracy for k={}: ".format(k), (cm[0][0]+cm[1][1])/cm['All']['All'])
print("TPR (Sensitivity) for k={}: ".format(k), cm[1][1]/cm[1]['All'])
print("FPR (false alarm ratio) for k={}: ".format(k), cm[1][0]/cm['All'][0])

#(a)vii.
print("\n (a)vii. Accuracy for k=5 was already reported in (a)vi.")

#(a)viii.
# Apply Zero-R algorithm
y_train = []
for review in traindata:
    y_train.append(int(review[0]))
y_train = pandas.Series(y_train)

if sum(y_train==0) <= sum(y_train==1):
    pred_zr = 1
else:
    pred_zr = 0
    
Nt = len(y_test)
y_pred_zr = [pred_zr]*Nt 
print("\n (a)viii. Accuracy using zero-r = {}%".format(100*sum(y_test==y_pred_zr)/Nt))


##########(c) ############
N = len(traindata)
cv_traindata = [traindata[:int(N/5)],
                          traindata[int(N/5):int(2*N/5)],
                          traindata[int(2*N/5):int(3*N/5)],
                          traindata[int(3*N/5):int(4*N/5)],
                          traindata[int(4*N/5):int(N)]]

y_test = []
for review in traindata:
    y_test.append(int(review[0]))
        
print("\n (c)i.")
for k in [3, 7, 99]:
    y_pred = []
    for cvset in range(5):
        y_pred.extend(customknn(k=k, traindata=list(itertools.chain.from_iterable(cv_traindata[:cvset-1]+cv_traindata[cvset:])), testdata=cv_traindata[-cvset], distance = '1/count'))
    
    y_pred = pandas.Series(y_pred)
    y_test = pandas.Series(y_test)

    cm = pandas.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    
    print("\n The CV accuracy for k = {} is: ".format(k), (cm[0][0]+cm[1][1])/cm['All']['All'])

###
y_test = []
for review in testdata:
    y_test.append(int(review[0]))
    
k=3

y_pred = customknn(k, traindata, testdata, distance = '1/count')

y_test = pandas.Series(y_test)
y_pred = pandas.Series(y_pred)
cm = pandas.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("\n (c)ii. Confusion matrix for k={}:\n\n".format(k), cm)
print("\n TN = ", cm[0][0])
print("FP = ", cm[1][0])
print("FN = ", cm[0][1])
print("TP = ", cm[1][1])
print("\n Accuracy for k={}: ".format(k), (cm[0][0]+cm[1][1])/cm['All']['All'])




########## (d) ###########
k=1

y_pred = customknn(k, traindata, testdata, distance = 'cosine')

y_test = pandas.Series(y_test)
y_pred = pandas.Series(y_pred)
cm = pandas.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("\n (d)iii. Confusion matrix for k={}:\n\n".format(k), cm)
print("\n TN = ", cm[0][0])
print("FP = ", cm[1][0])
print("FN = ", cm[0][1])
print("TP = ", cm[1][1])
print("\n (d)iv. Accuracy for k={}: ".format(k), (cm[0][0]+cm[1][1])/cm['All']['All'])
print("TPR (Sensitivity) for k={}: ".format(k), cm[1][1]/cm[1]['All'])
print("FPR (false alarm ratio) for k={}: ".format(k), cm[1][0]/cm['All'][0])


k=5

y_pred = customknn(k, traindata, testdata, distance = 'cosine')

y_test = pandas.Series(y_test)
y_pred = pandas.Series(y_pred)
cm = pandas.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("\n (d)v. Confusion matrix for k={}:\n\n".format(k), cm)
print("\n TN = ", cm[0][0])
print("FP = ", cm[1][0])
print("FN = ", cm[0][1])
print("TP = ", cm[1][1])
print("\n (d)vi. Accuracy for k={}: ".format(k), (cm[0][0]+cm[1][1])/cm['All']['All'])
print("TPR (Sensitivity) for k={}: ".format(k), cm[1][1]/cm[1]['All'])
print("FPR (false alarm ratio) for k={}: ".format(k), cm[1][0]/cm['All'][0])