"""
Created on Sun Dec 16 12:06:51 2018

@author: amc1354
"""

This programming assignment has been run using Python 3.6.5 distributed by Anaconda, so it is advisable to run with Python 3.6. Please make sure you have installed the packages used in the script, namely pandas, numpy and sklearn.

The python script to execute is named "RunMe.py", and it can be run simply by running on the terminal the command:
    
    python RunMe.py
    
assuming the command "python" prompts Python 3.6.
    
The scripts assume that the datasets "census_train.csv" and "census_test.csv" are in the same folder where the executable
script - namely "RunMe.py" - is.

Executing the script creates the required output as specified by the assignment on page 2: "Your final program must read in two files: a training file (census_train.csv) and a test file (census_test.csv). Your program must use the training file to learn a predictor, apply that predictor to the examples in the test file, and then write a file called test outputs.csv which gives the predictions for the unlabeled examples in the test file."

We did not include in "RunMe.py" all the rest of the analysis and the cross validation runs because it takes quite long to run depending on which laptop / system is run. The CV analysis and results included in the report and other parts not entirely reported for space constraints can be found in the file "analysis.py".

We tested the script in two different laptops both running Python 3.6. We did not test it with other distributions and 
versions of Python so, if there are any bugs, please do not hesitate to contact us at amc1354@nyu.edu or ads798@nyu.edu.

Thank you.