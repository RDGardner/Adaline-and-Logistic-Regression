#!/usr/bin/env python
# coding: utf-8

# # Programming Project 4
# ## 605.649 Introduction to Machine Learning
# ## Ricca Callis

# ## Directions
# 
# The purpose of this assignment is to give you a firm foundation in comparing a variety of linear classifiers.
# In this project, you will compare two different algorithms, one of which you have already implemented. These
# algorithms include Adaline and Logistic Regression. You will also use the same five datasets that you used
# from Project 1 from the UCI Machine Learning Repository, namely:
# 
# 1. Breast Cancer—https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%
# 29
# This breast cancer databases was obtained from the University of Wisconsin Hospitals, Madison from
# Dr. William H. Wolberg.
# 
# 2. Glass — https://archive.ics.uci.edu/ml/datasets/Glass+Identification
# The study of classification of types of glass was motivated by criminological investigation.
# 
# 3. Iris — https://archive.ics.uci.edu/ml/datasets/Iris
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
# 
# 4. Soybean (small) — https://archive.ics.uci.edu/ml/datasets/Soybean+%28Small%29
# A small subset of the original soybean database.
# 
# 5. Vote — https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
# This data set includes votes for each of the U.S. House of Representatives Congressmen on the 16 key
# votes identified by the Congressional Quarterly Almanac.
# 
# 
# When using these data sets, be careful of some issues.
# 
# 1. Not all of these data sets correspond to 2-class classification problems. A method for handling multiclass
# classification was described for Logistic Regression. For Adaline, it is suggested that you use what
# is called a “multi-net.” This is where you train a single network with multiple outputs. Note that if
# you wish to apply a one-vs-one or one-vs-all strategy for the neural network, that is acceptable. Just
# be sure to explain your strategy in your report.
# 
# 2. Some of the data sets have missing attribute values. When this occurs in low numbers, you may simply
# edit the corresponding values out of the data sets. For more occurrences, you should do some kind of
# “data imputation” where, basically, you generate a value of some kind. This can be purely random, or
# it can be sampled according to the conditional probability of the values occurring, given the underlying
# class for that example. The choice is yours, but be sure to document your choice.
# 
# 3. Most of attributes in the various data sets are either multi-value discrete (categorical) or real-valued.
# You will need to deal with this in some way. For the multi-value situation, you can apply what is called
# “one-hot coding” where you create a separate Boolean attribute for each value. For the continuous
# attributes, you may use one-hot-coding if you wish but, there is actually a better way. Specifically,
# it is recommended that you normalize them first to be in the range -1 to +1 and apply the inputs
# directly. (If you want to normalize to be in the range 0 to 1, that’s fine. Just be consistent.)

# ### For this project, the following steps are required:
# 
# For this project, the following steps are required:
# 
#  Download the five (5) data sets from the UCI Machine Learning repository. You can find this repository
# at http://archive.ics.uci.edu/ml/. All of the specific URLs are also provided above.
# 
#  Pre-process each data set as necessary to handle missing data and non-Boolean data (both classes and
# attributes).
# 
#  Implement Adaline and Logistic Regression.
# 
#  Run your algorithms on each of the data sets. These runs should be done with 5-fold cross-validation
# so you can compare your results statistically. You can use classification error, cross entropy loss, or
# mean squared error (as appropriate) for your loss function.
# 
#  Run your algorithms on each of the data sets. These runs should output the learned models in a
# way that can be interpreted by a human, and they should output the classifications on all of the test
# examples. If you are doing cross-validation, just output classifications for one fold each.
# 
#  Write a very brief paper that incorporates the following elements, summarizing the results of your
# experiments. Your paper is required to be at least 5 pages and no more than 10 pages using the JMLR
# format You can find templates for this format at http://www.jmlr.org/format/format.html. The
# format is also available within Overleaf.
# 
# 1. Title and author name
# 
# 2. Problem statement, including hypothesis, projecting how you expect each algorithm to perform
# 
# 3. Brief description of your experimental approach
# 
# 4. Presentation of the results of your experiments
# 
# 5. A discussion of the behavior of your algorithms, combined with any conclusions you can draw
# 
# 6. Summary
# 
# 7. References (Only required if you use a resource other than the course content.)
# 
#  Submit your fully documented code, the video demonstrating the running of your programs, and your
# paper.
# 
#  For the video, the following constitute minimal requirements that must be satisfied:
# 
#     – The video is to be no longer than 5 minutes long.
#     
#     – The video should be provided in mp4 format. Alternatively, it can be uploaded to a streaming
#     service such as YouTube with a link provided.
#     
#     – Fast forwarding is permitted through long computational cycles. Fast forwarding is not permitted
#     whenever there is a voice-over or when results are being presented.
#     
#     – Provide sample outputs from one test set showing classification performance on Adaline and
#     Logistic Regression
#     
#     – Show a sample trained Adaline model and Logistic Regression model
#     
#     – Demonstrate the weight updates for Adaline and Logistic Regression. For Logistic Regression,
#     show the multi-class case
#     
#     – Demonstrate the gradient calculation for Adaline and Logistic Regression. For Logistic Regression,
#     show the multi-class case
#     
#     – Show the average performance over the five folds for Adaline and Logistic Regression
# 
# Your grade will be broken down as follows:
# 
#  Code structure – 10%
# 
#  Code documentation/commenting – 10%
# 
#  Proper functioning of your code, as illustrated by a 5 minute video – 30%
# 
#  Summary paper – 50%

# In[1]:


# Author: Ricca Callis
# EN 605.649 Introduction to Machine Learning
# Programming Project #4
# Date Created: 7/20/2020
# File name: Programming Assignment 4 - Callis.ipynb
# Python Version: 3.7.5
# Jupyter Notebook: 6.0.1
# Description: Implementation of linear classifiers Adaline and Logistic Regression

"""
Adaline Algorithm: Parametric algorithm for classifying linearly separable classes. Attempts to predict the 
class value (0 or 1) using a linear transformation. Is trained using gradient descent.
"""

"""
Logistic Regression Algorithm: Parametric algorithm for classifying linearly seperable classes. 
Logistic Regression applies a log-odds transformation to the target class to make predictions on the 
likelihood of a instance belonging to a class. Is trained using gradient descent.
"""


"""
Required Data Sets:
    breast-cancer-wisconsin.data.csv
    breast-cancer-wisconsin.names.csv
    glass.data.csv
    glass.names.csv
    house-votes-84.data.csv
    house-votes-84.names.csv
    iris.data.csv
    iris.names.csv
    soybean-small.data.csv
    soybean-small.names.csv
""" 


# In[2]:


from platform import python_version
print ( python_version() )


# In[3]:


# Common standard libraries
import datetime
import time
import os
# Common external libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import sklearn #scikit-learn
import sklearn
from sklearn.model_selection import train_test_split 
import random as py_random
import numpy.random as np_random
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import scipy.stats as stats
from toolz import pipe # pass info from one process to another (one-way communication)
from typing import Callable, Dict, Union, List
from collections import Counter, OrderedDict
import logging
import multiprocessing
import operator
import sys
import copy
from typing import Callable, Dict, Union
from functools import partial
from itertools import product
import warnings
import io
import requests as r

#logging.basicConfig ( filename ='logfile.txt' )
logging.basicConfig()
logging.root.setLevel ( logging.INFO )
logger = logging.getLogger ( __name__ )

sys.setrecursionlimit ( 10000 )


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Check current directory
currentDirectory = os.getcwd()
print ( currentDirectory )


# In[6]:


# Input data files are available in the ".../input/" directory
# Change the Current working Directory
os.chdir ( '/Users/riccacallis/Desktop/JHU/Data Science/Introduction to Machine Learning/Programming Project 4/input' )

# Get Current working Directory
currentDirectory = os.getcwd()
print ( currentDirectory )


# In[7]:


# List files in input directory
from subprocess import check_output
print ( check_output ( [ "ls", "../input" ] ).decode ( "utf8" ) )


# # Adaline (Adaptive Linear Neuron)
# 
# **Overview:**
# 
# A network model proposed by Bernard Widrow in 1959.
# 
# Used for binary classification tasks.
# 
# ![adaline1.png](attachment:adaline1.png)
# 
# 
# ## The Training Rule
# 
# The activation function used is:
# 
#     y = 1 if y_in ≥ 0
#     
#     y = -1 if y_in < 0
#     
# 
# The training rule is called the Widrow-Hoff rule or the Delta Rule
# 
# It can be theoretically  shown that the rule minimizes the root mean square error between the ac$va$on value and the target value.
# 
# That’s why it’s called the the Least Mean Square (LMS) rule as well.
# 
# ## The δ Rule
# 
# The δ rule works also works for more than one output unit.
# 
# Consider one single output unit.
# 
# The delta rule changes the weights of the neural connections so as to minimize the difference between the net input to the output unit y_in and the target value t.
# 
# The goal is to minimize the error over all training patterns.
# 
# However, this is accomplished by reducing the error to each pattern one at a time.
# 
# Weight corrections can also be accumulated over a number of training patterns (called batch updating) if desired.
# 
# 
# ### The Training Algorithm
# 
# 
# Initialize weights to small random values
# 
# Set learning rate α to a value between 0 and 1 
# 
# while (the largest weight change ≤ threshold) do
# 
#     for each bipolar training pair s:t do
#     
#         {Set activation of input units i=1..n {xi = si}
#         
#         Compute net input to the output unit:
#         
#             y_in = b + Σx_i w_i 
#         
#         Update bias and weights:
#         
#             for i=1..n {
#             
#                 b(new) = b(old) + α(t – y_in) w_i(new) = w_i(old) + α (t – y_in)x_i}
#                 } //endfor 
# }//end while
# 
# ### Setting Learning Parameter α
# 
# Usually, just use a small value for α, something like 0.1.
# 
# If the value is too large, the learning process will not converge.
# 
# If the value of α is too small, learning will be extremely slow (Hecht-Nielsen 1990).
# 
# For a single neuron, a prac$cal range for α is 0.1 ≤ n × α ≤ 1.0, where n is the number of input units (Widrow, Winger and Baxter 1988).

# ## MADALINE (Many Adalines)
# 
# When several ADALINE units are arranged in a single layer so that there are several output units, there is no change in how ADALINEs are trained from that of a single ADALINE.
# 
# A MADALINE consists of many ADALINEs arranged in a multilayer net.
# 
# We can think of a MADALINE as having a hidden layer of ADALINEs.
#     
#     • A Madaline is composed of several Adalines
#     
#         • Each ADALINE unit has a bias. 
#         
#         • There are two hidden ADALINEs, z1 and z2.
#         
#         • There is a single output ADALINE Y.
#         
#     • Each ADALINE simply applies a threshold function to the unit’s net input.
# 
# Y is a non-linear function of the input vector (x1, x2).
# 
# The use of hidden units Z1 and Z2 gives the net additional power, but makes training more complicated.
# 
# ### Madaline Training 
# 
# There are two training algorithms for a MADALINE with one hidden layer.
# 
#     • Algorithm MR-I is the original MADALINE training algorithm (Widrow and Hoff 1960).
#         
#         - MR-I changes the weights on to the hidden ADALINEs only. 
#         
#         - The weights for the output unit are fixed. 
#         
#         - It assumes that the output unit is an OR unit.
#         
# 
#     • MR-II (Widrow, Winter and Baxter 1987) adjusts all weights in the net. 
#     
#         - It doesn’t make the assumption that the output unit is an OR unit.

# In[8]:


# Implementation of Adaline Algorithm

'''''
Class: Adaline
    - Parametric Machine Learning algorithm for classifying linearly separable classes. Attempts to predict 
    the class value (0 or 1) using a linear transformation. Is trained using gradient descent.

Functions:
    - __init__: Initializes the Adaline algorithm
    - get_updated_gradient_term: Function which obtains the updated gradient term.
    - fit: Fitting procedure for Logistic Regression
    - predict_probabilities: Method which obtains the output scores for predictions on feature matrix, X.
    - predict: Makes predictions on feature matrix, X.
'''''

class Adaline:
    """
    Class to fit Adaline (Adaptive Linear Neuron)
    """

    def __init__(
        # Initialize parameters
        self, # class instance
        convergence_tolerance = 0.1, # Stopping criteria based on value of gradient
        learning_rate = 0.01, # ETA for gradient update
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 5, # Stopping criteria based on number of updates
    ):
        """
         Parameters:
             convergence_tolerance: Indicates the stopping criteria based on the value of the gradient. (Type: float)
             learning_rate: ETA for the gradient update. (Type: float) 
             fit_intercept: If true, will add column of 1s. (Type: boolean)
             max_iter: Indicates the stopping criteria based on the number of updates. (Type: Integer)
         """
        # Initialize class instances
        self.convergence_tolerance = convergence_tolerance
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    @staticmethod
    def get_updated_gradient_term ( feature_matrix, target_array, weights ):
        """
        Method which obtains the updated gradient term.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                weights: Indicates the weights of the logistic regression (Type: Array; np.ndarray)
            Returns: Updated gradient term, calculated as (weights in X) - y
        """
        # Return updated gradient term
        return ( ( weights @ feature_matrix.T ) - target_array) @ feature_matrix
        # End get_updated_gradient_term()

    def fit ( self, feature_matrix, target_array ):
        """
        Method fits a logistic regression (LogisticRegression).
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                weights: Indicates the weights of the logistic regression (Type: Array; np.ndarray)
        """
        # Add column of ones
        if self.fit_intercept:
            feature_matrix = np.concatenate ( [ np.ones ( ( feature_matrix.shape [ 0 ] , 1 ) ), feature_matrix ], axis = 1 )

        # Initialize random weights
        self.weights = np.random.uniform ( low = -0.01, high = 0.01, size = ( 1, feature_matrix.shape [ 1 ] ) )

        # Get the gradient
        gradient = Adaline.get_updated_gradient_term (
            feature_matrix = feature_matrix, target_array = target_array, weights = self.weights
        )

        iter_count = 1
        
        # While convergence criteria not met
        while (
            np.any ( np.abs ( gradient ) > self.convergence_tolerance )
            and iter_count < self.max_iter
        ):
            # Update weights
            self.weights = self.weights - self.learning_rate * gradient

            # Calculate weights
            gradient = Adaline.get_updated_gradient_term (
                feature_matrix = feature_matrix, target_array = target_array, weights = self.weights
            )

            # Increment count
            iter_count += 1

            # Stop if gradient explodes
            if pd.isnull ( gradient ).any():

                print ( "Exploding gradient" )
                break
        # End fit()

    def predict_probabilities ( self, feature_matrix ):
        """
        Method which obtains the output scores for predictions on feature matrix, X.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
            Returns: Array of output scores for predictions on feature matrix, X.
        """
        # Add ones if fit_intercept
        if self.fit_intercept:
            feature_matrix = np.concatenate ( [ np.ones ( ( feature_matrix.shape [ 0 ], 1 ) ), feature_matrix ], axis = 1 )
        # Make predictions
        return ( self.weights @ feature_matrix.T ).reshape ( -1 )

    def predict ( self, feature_matrix ):
        """
        Method which makes predictions on feature matrix, X.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
            Returns: Array of 0s or 1s. 
        """
        # Round to 0 or 1
        return np.round ( self.predict_probabilities ( feature_matrix = feature_matrix ) )
        # End predict()
# End class Adaline

def mode ( target_array ):
    """
    Method to obtain the mode (i.e., most common value) of the target array, y.
        Parameters:
            target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
        Returns: Mode (i.e., most common value) of the target array, y.    
    """
    # Return most common value (i.e., mode)
    return Counter ( target_array ).most_common ( 1 ) [ 0 ] [ 0 ]
    # End mode()

'''''
Class: MaxScaler
    - Class that scales everything to [-1, 1] interval.

Functions:
    - fit: Obtains the maximum values in the feature matrix, X.
    - transform: Scales the feature matrix, X by the max values.
    - fit_transform: Adjusts the feature matrix, X, to [-1, 1] interval
'''''
class MaxScaler:
    """
    Class that scales everything to [-1, 1] interval.
    """

    def fit ( self, feature_matrix ):
        """
        Method to obtain the maximum values in the feature matrix, X.
            Parameters:
                self: Indicates the class instance.
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)   
        """
        # Get the max values
        self.maxes = np.abs ( feature_matrix ).max()
        # End fit ()

    def transform ( self, feature_matrix ):
        """
        Method to scale the feature matrix, X,  by the max values
            Parameters:
                self: Indicates the class instance.
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)   
        """
        # Scale by said values
        return feature_matrix / self.maxes
        # End transform()

    def fit_transform ( self, feature_matrix ):
        """
        Method to scale the feature matrix, X, to the [-1, 1] interval
            Parameters:
                self: Indicates the class instance.
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)   
        """
        self.fit ( feature_matrix )
        return self.transform ( feature_matrix )
        # End fit_transform()
# End class MaxScaler


# # Logistic Regression
# 
# 
# **Overview:**
# 
# a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
# 
# In logistic regression, we’re essentially trying to find the weights that maximize the likelihood of producing our given data and use them to categorize the response variable.
# 
# Since the likelihood maximization in logistic regression doesn’t have a closed form solution, we’ll solve the optimization problem with gradient descent. 
# 
# ## Assumptions
# 
# Binary logistic regression requires the dependent variable to be binary.
# 
# For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.
# Only the meaningful variables should be included.
# 
# The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
# 
# The independent variables are linearly related to the log odds.
# 
# Logistic regression requires quite large sample sizes.
# 
# ## Logistic Function
# 
# 
# ![graph%20of%20logistic%20function.png](attachment:graph%20of%20logistic%20function.png)
# 
# The logistic function σ(z) is an S-shaped curve defined as:
# 
# ![logistic%20function%20defined.png](attachment:logistic%20function%20defined.png)
# 
# It is also sometime known as the expit function or the sigmoid. 
# 
# It is monotonic and is bounded between 0 and 1, hence its widespread usage as a model for a probability. We moreover have:
# ![logistic%20function%20monotonic.png](attachment:logistic%20function%20monotonic.png)
# 
# Finally, you can easily show that its derivative with respect to z is given by:
# ![logistic%20function%20derivative.png](attachment:logistic%20function%20derivative.png)
# 
# 
# ### Binary Cross-Entropy
# 
# Logistic Regression relies on the minimization of the binary cross-entropy:
# 
# ![binary%20cross-entropy.png](attachment:binary%20cross-entropy.png)
# 
#     where,
#         m is the number of samples
#         
#         xᵢ is the i-th training example
#         
#         yᵢ its class (i.e. either 0 or 1)
#         
#         σ(z) is the logistic function
#         
#         w is the vector of parameters of the model.
# 
# **Deriving The Binary Cross-Entropy**
# 
# Let us consider a predictor x and a binary (or Bernoulli) variable y. Assuming there exist some relationship between x and y, an ideal model would predict:
# 
# ![Bernoulli.png](attachment:Bernoulli.png)
# 
# By using logistic regression, this unknown probability function is modeled as:
# 
# ![Probability%20Function.png](attachment:Probability%20Function.png)
# 
# Our goal is thus to find the parameters w such that the modeled probability function is as close as possible to the true one.
# 
# **From the Bernoulli distribution to the binary cross-entropy**
# 
# One way to assess how good of a job our model is doing is to compute the so-called likelihood function. Given m examples, this likelihood function is defined as:
# ![Likelihood%20Function.png](attachment:Likelihood%20Function.png)
# 
# Ideally, we thus want to find the parameters w that maximize ℒ(w). In practice however, one usually does not work directly with this function but with its negative log for the sake of simplicity:
# ![Negative%20Log.png](attachment:Negative%20Log.png)
# 
# Because logarithm is a strictly monotonic function, minimizing the negative log-likelihood will result in the same parameters w as when maximizing directly the likelihood function. But how to compute P(y|x, w) when our logistic regression only models P(1|x, w) ? Given that
# 
# ![Given.png](attachment:Given.png)
# 
# one can use a simple exponentiation trick to write
# 
# ![exponentiation.png](attachment:exponentiation.png)
# 
# Inserting this expression into the negative log-likelihood function (and normalizing by the number of examples), we finally obtain the desired normalized binary cross-entropy
# 
# ![normalized%20binary%20cross-entropy.png](attachment:normalized%20binary%20cross-entropy.png)
# 
# Finding the weights w minimizing the binary cross-entropy is thus equivalent to finding the weights that maximize the likelihood function assessing how good of a job our logistic regression model is doing at approximating the true probability distribution of our Bernoulli variable.

# ### Efficiency
# Unlike linear regression, no closed-form solution exists for logistic regression. Since the binary cross-entropy is a convex function, any technique from convex optimization is guaranteed to find the global minimum. 
# 
# Here, we’ll use gradient descent with optimal learning rate.
# 
# **Gradient descent with optimal learning rate**
# 
# In gradient descent, the weights w are iteratively updated following the simple rule:
# 
# ![gradient%20descent%20rule.png](attachment:gradient%20descent%20rule.png)
# 
# until convergence is reached. Here, α is known as the learning rate or step size. It is quite common to use a constant learning rate but how to choose it ? By computing the expression of the Lipschitz constant of various loss functions, Yedida & Saha have recently shown that, for the logistic regression, the optimal learning rate is given by:
# 
# ![optimal%20learning%20rate.png](attachment:optimal%20learning%20rate.png)
# 

# In[9]:


# Implementation of Logistic Regression Classifier 

'''''
Class: LogisticRegression
    - Parametric Machine Learning algorithm for classifying linearly separable classes. Attempts to predict 
    the class value (0 or 1) using a linear transformation. Is trained using gradient descent.

Functions:
    - __init__: Initializes the LogisticRegression algorithm
    - get_matrix_product: Method which obtains the weights of the input matrix/feature matrix, X.
    - get_class_scores: Method which obtains the class scores for each of the classes.
    - get_updated_gradient_term: Function which obtains the updated gradient term.
    - fit: Fitting procedure for Logistic Regression
    - predict_probabilities: Method which obtains the output scores for predictions on feature matrix, X.
    - predict: Makes predictions on feature matrix, X.
'''''
class LogisticRegression:
    """
    Class to fit a logistic regression classifier
    """

    def __init__(
         # Initialize parameters
        self, # Class instance
        convergence_tolerance = 0.1, # Stopping criteria based on value of gradient
        learning_rate = 0.01, # ETA for gradient update
        fit_intercept = True,  # If true, will add column of 1s
        max_iter = 5, # Stopping criteria based on number of updates
    ):
        """
         Parameters:
             convergence_tolerance: Indicates the stopping criteria based on the value of the gradient. (Type: float)
             learning_rate: ETA for the gradient update. (Type: float) 
             fit_intercept: If true, will add column of 1s. (Type: boolean)
             max_iter: Indicates the stopping criteria based on the number of updates. (Type: Integer)
        """
        self.fit_intercept = fit_intercept
        self.convergence_tolerance = convergence_tolerance
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    @staticmethod
    def get_matrix_product ( feature_matrix, weights ):
        # Weights by feature_matrix, "X"
        return weights @ feature_matrix.T
        # End get_matrix_product()

    @staticmethod
    def get_class_scores ( feature_matrix, weights ):
        """
        Method which obtains the class scores for each of the classes.
            Parameters:
                feature_matrix: Indicates the input matrix/feature matrix, "X". (Type: Array; np.ndarray)
                weights: Indicates the weights of the linear transformation. (Type: Array; np.ndarray)
            Returns: Class scores for each feature in the input matrix 
            (or, the normalized likelihopod for all the classes).
        """
        matrix_product = LogisticRegression.get_matrix_product (
            feature_matrix = feature_matrix, weights = weights
        )

        # Get the normalized likelihood for all the classes
        return ( np.exp ( matrix_product ) / np.sum ( np.exp ( matrix_product ), axis = 0 ) ).T
        # End get_class_scores()

    @staticmethod
    def get_updated_gradient_term ( feature_matrix, target_array, weights ):
        """
        Method which obtains the updated gradient term.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                weights: Indicates the weights of the logistic regression (Type: Array; np.ndarray)
            Returns: Updated gradient term
        """
        class_scores = LogisticRegression.get_class_scores (
            feature_matrix = feature_matrix, weights = weights
        )

        # Get the one-hots for y classes
        y_one_hot = pd.get_dummies ( target_array ).values

        # Get the gradient (r - y) X
        return np.dot ( ( class_scores - y_one_hot ).T, feature_matrix )
        # End get_updated_gradient_term()

    def fit ( self, feature_matrix, target_array ):
        """
        Method which fits the logistic regression (LogisticRegression).
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                weights: Indicates the weights of the logistic regression (Type: Array; np.ndarray)
        """
        # Add column of ones if fit_intercept
        if self.fit_intercept:
            feature_matrix = np.concatenate ( [ np.ones ( ( feature_matrix.shape [ 0 ], 1 ) ), feature_matrix ], axis = 1 )

        # Get the classes
        classes = set ( target_array )

        # Initialize random weights around 0
        self.weights = np.random.uniform (
            low = -0.01, high = 0.01, size = ( len ( classes ), feature_matrix.shape [ 1 ] )
        )

        # Calculate the gradient
        gradient = LogisticRegression.get_updated_gradient_term (
            feature_matrix = feature_matrix, target_array = target_array, weights = self.weights
        )

        iter_count = 1

        # While convergence criteria not met
        while (
            np.any ( np.abs ( gradient ) > self.convergence_tolerance )
            and iter_count < self.max_iter
        ):
            # Update weights
            self.weights = self.weights - self.learning_rate * gradient

            # Calculate weights
            gradient = LogisticRegression.get_updated_gradient_term (
                feature_matrix = feature_matrix, target_array = target_array, weights = self.weights
            )

            # Increment count
            iter_count += 1

            # Stop if gradient explodes
            if pd.isnull ( gradient ).any():
                print ( "Exploding gradient" )
                break
        # End fit()

    def predict_probabilities ( self, feature_matrix ):
        """
        Method which obtains the output scores for predictions on feature matrix, X.
            Parameters:
                feature_matrix: Indicates the input matrix/feature matrix, "X". (Type: Array; np.ndarray)
            Returns: Array of output scores for predictions on feature matrix, X.
        """
        
        # Add ones if fit_intercept
        if self.fit_intercept:
            feature_matrix = np.concatenate ( [ np.ones ( ( feature_matrix.shape [ 0 ], 1 ) ), feature_matrix ], axis = 1 )
        # Make predictions
        return self.weights @ feature_matrix.T
        # End predict_probabilities()

    def predict ( self, feature_matrix ):
        """
        Method which makes predictions on feature matrix, "X".
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
        """

        # Get the largest class prediction
        return np.argmax ( self.predict_probabilities ( feature_matrix ), axis = 0 )
        # End predict()


# # Model Evaluation
# 
# Loss functions are used by algorithms to learn the classification models from the data.
# 
# Classification metrics, however, evaluate the classification models themselves. 
# 
# For a binary classification task, where "1" is taken to mean "positive" or "in the class" and "0" is taken to be "negative" or "not in the class", the cases are:
# 
# 1. The true class can be "1" and the model can predict "1". This is a *true positive* or TP.
# 2. The true class can be "1" and the model can predict "0". This is a *false negative* or FN.
# 3. The true class can be "0" and the model can predict "1". This is a *false positive* or FP.
# 4. The true class can be "0" and the model can predict "0". This is a *true negative* or TN.
# 

# ## Training Learners with Cross-Validation
# 
# Fundamental assumption of machine learning:The data that you train your model on must come from the same distribution as the data you hope to apply the model to.
# 
# Cross validation is the process of training learners using one set of data and testing it using a different set.
# 
# Options:
#     - Divide your data into two sets:
#         1. The training set which you use to build the model
#         2. The test(ing) set which you use to evaluate the model. 
#     - kfolds: Yields multiple estimates of evaluation metric
# 
#     
# ### k-fold Cross-Validation
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
# 
# The procedure has a single parameter called k that refers to the number of groups (or folds) that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=5 becoming 5-fold cross-validation.
# 
# 
# The general procedure is as follows:
# - Shuffle the dataset randomly.
# - Split the dataset into k groups (or folds)
# - Save first fold as the validation set & fit the method on the remaining k-1 folds
# - For each unique group:
#     - Take the group as a hold out or test data set
#     - Take the remaining groups as a training data set
# - Fit a model on the training set and evaluate it on the test set
# - Retain the evaluation score and discard the model
# - Summarize the skill of the model using the sample of model evaluation scores
#     - The average of your k recorded errors is called the cross-validation error and will serve as your performance metric for the model
# 
# Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.
# 
# Below is the visualization of a k-fold validation when k=10.
# 
# Looks like:
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | **Test** | Train | Train | Train | Train | Train | Train | Train | Train | Train |
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| **Test** | Train | Train | Train | Train | Train | Train | Train | Train |
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| Train | **Test** | Train | Train | Train | Train | Train | Train | Train |
# 
# And finally:
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| Train | Train | Train | Train | Train | Train | Train | Train | **Test** |
# 
# ### Stratified k-fold Cross-Validation
# Stratification is the process of rearranging the data so as to ensure that each fold is a good representative of the whole. For example, in a binary classification problem where each class comprises of 50% of the data, it is best to arrange the data such that in every fold, each class comprises of about half the instances.
# 
# For classification problems, one typically uses stratified k-fold cross-validation, in which the folds are selected so that each fold contains roughly the same proportions of class labels.
# 

# In[10]:


# Model Evaluation

# Teaching Learners with Cross-Validation
# k-Folds

'''''
Class: KFoldStratifiedCV
    - Class to conduct Stratified K-Fold Cross Validation. Ensures the splitting of data into folds is governed by 
    criteria such as ensuring that each fold has the same proportion of observations with a given categorical 
    value, such as the class outcome value.

Functions:
    - __init__: Initializes the KFoldStratifiedCV algorithm 
    - add_split_col: Adds new column called "split"
    - split: Takes an array of classes, and creates train/test splits with proportional examples for each group.
'''''

class KFoldStratifiedCV:
    """
    Class to conduct Stratified K-Fold Cross Validation.
        Parameters
            number_of_folds: Indicates the number of folds or splits. Type: Integer
            
    """

    def __init__( self, number_of_folds, shuffle = True ):
        # Initialize parameters
        # Class Instances
        self.number_of_folds = number_of_folds
        self.shuffle = shuffle

    def add_split_col ( self, feature_array ):
        """
        Function adds new column called "split"
            Parameters
                feature_array: Indicates the feature array
            Returns: New column in dataframe with index & split 
        """
        feature_array = feature_array if not self.shuffle else np.random.permutation ( feature_array )
        n = len ( feature_array )
        k = int ( np.ceil ( n / self.number_of_folds ) )
        return pd.DataFrame (
            { "idx": feature_array, "split": np.tile ( np.arange ( self.number_of_folds ), k )[ 0 : n ] , }
        )
        # End add_split_col

    def split ( self, target_array, feature_matrix = None ):
        """
        Function takes an array of classes, and creates train/test splits with proportional examples for each group.
            Parameters
                target_array: Indicates the array of class labels. Type: Array (np.array)
            Returns: Dataframe with index values of not cv split & cv split train and test data
        """
        # Make sure y is an array
        target_array = np.array ( target_array ) if isinstance ( target_array, list ) else target_array

        # Groupby y and add integer indices.
        df_with_split = (
            pd.DataFrame ( { "y": target_array, "idx": np.arange ( len ( target_array ) ) } )
            .groupby ( "y" ) [ "idx" ]
            .apply ( self.add_split_col )  # Add col for split for instance
        )

        # For each fold, get train and test indices (based on col for split)
        for cv_split in np.arange ( self.number_of_folds - 1, - 1, - 1 ):
            train_bool = df_with_split [ "split" ] != cv_split
            test_bool = ~ train_bool
            # Yield index values of not cv_split and cv_split for train, test
            yield df_with_split [ "idx" ].values [ train_bool.values ], df_with_split [
                "idx"
            ].values [ test_bool.values ]
        # End split()
# End class KFoldStratifiedCV

"""
This part is the code for a one vs. rest classifier 
"""


'''''
Class: MulticlassClassifier
    - Class to do one vs. rest multiclass classification using Boolean output classifier.

Functions:
    - __init__: Initializes the KFoldStratifiedCV algorithm 
    - _get_y_binary: Transforms multivalued outputs into one vs. rest booleans (where class = 1)
    - fit: Fits the classifiers across all the models.
    - predict: Gets the highest probability class across all the one vs. rest classifiers.
'''''

class MulticlassClassifier:
    """
    Class to do one vs. rest multiclass classification using Boolean output classifier.
        Parameters
            model_class : Indicates a callable that returns the model object to use in fitting (Type: Callable).
            classes : np.ndarray Indicates an array containing the values in `y` (which are uysed to create a classifier).
            class_kwargs : A dictionary of args for `model_class` mapping the class value to a dictionary of kwargs. (Type: Dictionary)
    """

    def __init__ ( self, model_class: Callable, classes: np.ndarray, class_kwargs):

        self.classes = classes
        # Create the models (mapping from class to model)
        self.models = {
            element: model_class( ** class_kwargs.get ( element ) ) for element in self.classes
        }

    @staticmethod
    def _get_y_binary ( target_array, cls):
        """
        Method which transforms the multivalued outputs into one vs rest booleans
            Parameters:
                target_array: Indicates the array of class labels. (Type: Array; np.array)
                class: Indicates the class' classification
            Returns: Either 0 or 1, where 1 indicates class membership
        """
        # Transform multivalued outputs into one vs. rest booleans
        # where `cls` is the value of 1.
        return np.where ( target_array == cls, 1, 0 )
        # End _get_y_binary()

    def fit ( self, feature_matrix, target_array ):
        """
        Fit the classifiers across all the models.
            Parameters:
                feature_matrix: Indicates the input matrix/feature matrix, "X". (Type: Array; np.array)
                target_array: Indicates the array of class labels. (Type: Array; np.array)
        """
        if set ( target_array ) - set ( self.classes ):
            raise ValueError ( "y contains elements not in `classes`" )

        for cls, model in self.models.items():
            # Create the binary response for `cls`
            y_binary = MulticlassClassifier._get_y_binary ( target_array, cls )
            # Fit the the model for that class.
            model.fit ( feature_matrix, y_binary )
        # End fit()

    def predict ( self, feature_matrix ):
        """
        Gets the highest probability class across all the one vs. rest classifiers.
            Parameters:
                feature_matrix: Indicates the input matrix/feature matrix, "X". (Type: Array; np.array)
            Returns: The class corresponding to the largest probability
        """
        # Get the prediction_prob across all the classes.
        predictions = { cls: model.predict_probabilities ( feature_matrix ) for cls, model in self.models.items() }

        # Get the class corresponding to the largest probability.
        return [
            max ( predictions.keys(), key = lambda x: predictions [ x ] [ prediction ] )
            for prediction in range ( feature_matrix.shape [ 0 ] )
        ]
        # End predict()


# # Breast Cancer Data Set
# ## Extract, Transform, Load: Breast Cancer Data
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# 
# ### Attribute Information: 10 Attributes (d)
# 
# 1. Sample code number: id number 
# 2. Clump Thickness: 1 - 10 
# 3. Uniformity of Cell Size: 1 - 10 
# 4. Uniformity of Cell Shape: 1 - 10 
# 5. Marginal Adhesion: 1 - 10 
# 6. Single Epithelial Cell Size: 1 - 10 
# 7. Bare Nuclei: 1 - 10 
# 8. Bland Chromatin: 1 - 10 
# 9. Normal Nucleoli: 1 - 10 
# 10. Mitoses: 1 - 10
# 
# ### One Class Label
# 11. Class: (2 for benign, 4 for malignant)

# In[11]:


if __name__ == "__main__":
    np.random.seed ( 7202020 ) # Today's Date

# Log ETL: Breast Cancer Data
logger.info ( "ETL: Breast Cancer Data Set" )

# Read Breast Cancer Data
cancer_data = (
    pd.read_csv(
        "breast-cancer-wisconsin.data.csv",
        header = None,
        #Assign labels to columns
        names = [
            "id_number",
            "clump_thickness",
            "uniformity_cell_size",
            "uniformity_cell_shape",
            "marginal_adhesion",
            "single_epithelial_cell_size",
            "bare_nuclei",
            "bland_chromatin",
            "normal_nucleoli",
            "mitosis",
            "class",
        ],
    )
    .replace ( "?", np.NaN ) # Remove all instances with "?"
    .dropna ( axis = 0, how = "any" ) # Delete all NaN
    .astype ( "int" ) # Assign all values as type: integer
)


# In[12]:


# Confirm data was properly read by examining data frame
cancer_data.info()


# **Notes** 
# 
# As expected, Dataframe shows us we have 683 total observations & 11 columns (or, 11 variables). Each variable has 683 observations, so we know we have no missing data. 
# 
# We see that all columns are integers, just as we want it

# In[13]:


# Log ETL
logger.info ( f"ETL, Dataframe Info: { cancer_data.head() }" )


# In[14]:


# Look at first few rows of dataframe
cancer_data.head()


# In[15]:


# Verify whether any values are null
cancer_data.isnull().values.any()


# In[16]:


# Again
cancer_data.isna().any()


# In[17]:


# Drop id_number from cancer data set
cancer_data.drop ( [ 'id_number' ], axis = 'columns', inplace = True )
# Confirm
cancer_data.head()


# In[18]:


# Map Current class values
# Now, 2 = Nonmembership (benign); 4 = Membership (malignant)
# We want 2 = 0 and 4 = 1
cancer_data [ 'class' ]= cancer_data [ 'class' ].map ( { 2 : 0, 4 : 1 } )
# Confirm
cancer_data.head()


# In[19]:


# Classification for Class Label: data frame for this category
cancer_data[ "class" ].astype ( "category" ).cat.codes


# In[20]:


# Confirm
cancer_data.head()


# In[21]:


# One-hot encoding/Dummy Variables for remaining cancer data
cancer_data_boolean = pipe (cancer_data,lambda df: pd.get_dummies ( data = df, columns = [ col for col in df.columns if col != "class" ],
        drop_first = True ) )
# Confirm
cancer_data_boolean.head()


# ## (Brief) Exploratory Data Analysis: Breast Cancer Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5

# In[22]:


# Log EDA: Breast Cancer Data
logger.info ( "EDA: Breast Cancer Data Set" )


# In[23]:


# Descriptive Statistics
cancer_data.describe()


# **Notes**
# 
# Total number of observations: 683
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 
# If we wanted, we could use this information for each attribute to calculate the following:
#    - Interquartile Range: Q3-Q1
#    - Whisker: 1.5 * IQR (Outliers lie beyond the whisker)

# In[24]:


# Log Descriptives
logger.info ( f"Descriptive Statistics: { cancer_data.describe() }" )


# ## (Brief) Exploratory Data Analysis: Breast Cancer Data
# 
# ### Pair-Wise: Attribute by Class

# In[25]:


# Frequency of diagnoses classifications
cancer_data [ 'class' ].value_counts() # raw counts


# **Notes**
# 
# Class = 0 (benign) has 444 (out of 683) instances (~65.01%)
# 
# Class = 1 (malignant) has 239 (out of 683) instances (~34.99%)

# In[26]:


#Log Pair-Wise Attribute by Class
logger.info ( f"Pair-Wise Attribute by Class: { cancer_data [ 'class' ].value_counts() }" )


# In[27]:


# Plot diagnosis frequencies
sns.countplot ( cancer_data [ 'class' ],label = "Count" ) # boxplot


# In[28]:


def describe_by_category ( data, numeric, categorical, transpose = False ):
    grouped = data.groupby ( categorical )
    grouped_y = grouped [ numeric ].describe()
    if transpose:
        print( grouped_y.transpose() )
    else:
        print ( grouped_y )


# In[29]:


# Descriptive Statistics: Describe each variable by class (means only)
cancer_data.groupby ( [ 'class' ] )[ 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape', 'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitosis' ].mean()


# In[30]:


# Descriptive Statistics: Describe each variable by class
cancer_data.groupby ( [ 'class' ] )[ 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape', 'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitosis' ].describe()


# In[31]:


boxplot = cancer_data.boxplot ( column = [ 'clump_thickness', 'uniformity_cell_size' ], by = [ 'class' ] )


# In[32]:


boxplot = cancer_data.boxplot ( column = [ 'uniformity_cell_shape', 'marginal_adhesion' ], by = [ 'class' ] )


# In[33]:


boxplot = cancer_data.boxplot ( column = ['single_epithelial_cell_size', 'bare_nuclei'], by = [ 'class' ] )


# In[34]:


boxplot = cancer_data.boxplot ( column = [ 'bland_chromatin', 'normal_nucleoli', 'mitosis'  ], by = [ 'class' ] )


# In[35]:


# Descriptive Statistics: Clump Thickness by Class
describe_by_category ( cancer_data, "clump_thickness", "class", transpose = True )


# In[36]:


# Descriptive Statistics: Uniformity Cell Size by Class
describe_by_category ( cancer_data, "uniformity_cell_size", "class", transpose = True )


# In[37]:


# Descriptive Statistics: Uniformity Cell Shape by Class
describe_by_category ( cancer_data, "uniformity_cell_shape", "class", transpose = True )


# In[38]:


# Descriptive Statistics: Marginal Adhesion by Class
describe_by_category ( cancer_data, "marginal_adhesion", "class", transpose = True )


# In[39]:


# Descriptive Statistics: Single Epithelial Cell Size by Class
describe_by_category ( cancer_data, "single_epithelial_cell_size", "class", transpose = True )


# In[40]:


# Descriptive Statistics: Bare Nuclei by Class
describe_by_category ( cancer_data, "bare_nuclei", "class", transpose = True )


# In[41]:


# Descriptive Statistics: Bland Chromatin by Class
describe_by_category ( cancer_data, "bland_chromatin", "class", transpose = True )


# In[42]:


# Descriptive Statistics: Normal Nucleoli by Class
describe_by_category ( cancer_data, "normal_nucleoli", "class", transpose = True )


# In[43]:


# Descriptive Statistics: Mitosis by Class
describe_by_category ( cancer_data, "mitosis", "class", transpose = True )


# ## Linear Classification Experiments: Breast Cancer Data
# 
# ### Assign Feature Matrix & Target Vector

# In[44]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Array; Categorical instance class (i.e., doesn't include the attribute features)

ms = MaxScaler()

feature_matrix, target_array = (
        cancer_data.drop ( [ "class"], axis = 1 ).values,
        cancer_data [ "class" ].astype ( "category" ).cat.codes.values,
    )


# ### Run Linear Classification
# ### Adaline & Logistic Regression 
# ### Stratified Cross-Fold Validation

# In[45]:


# Experiment: Breast Cancer Data

# Log Experiment: Running Linear Classification Experiment using Adaline & Logistic Regression Algorithms on Breast Cancer Data
logger.info ( "Running Breast Cancer Linear Classification Experiment: Adaline & Logistic Regression" )

# Use 5 folds for cross-validation
kfold = KFoldStratifiedCV ( number_of_folds = 5 )
accuracy_adaline = []
accuracy_lr = []
baseline = []

# Stratifier KFOLD CV
for train, test in kfold.split ( feature_matrix = feature_matrix, target_array = target_array ):
    # Adaline
    adaline = Adaline (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 100, # Stopping criteria based on number of updates
        learning_rate = 0.0001, # ETA for gradient update
    )
    
    # Logistic Regression
    logistic_regression = LogisticRegression (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 100, # Stopping criteria based on number of updates
        learning_rate = 0.0001, # ETA for gradient update
    )

    # Append the baseline accuracy (mode of training target)
    baseline.append ( np.mean ( mode ( target_array [ train ] ) == target_array [ test ] ) )

    # Fit the models on the transformed training data
    #Fit Adaline
    adaline.fit ( ms.fit_transform ( feature_matrix [ train ] ), target_array [ train ] )
    #Fit Logistic Regression
    logistic_regression.fit ( ms.fit_transform ( feature_matrix [ train ] ), target_array [ train ] )

    # Append the accuracy
    # Append Adaline accuracy
    accuracy_adaline.append (
        np.mean ( adaline.predict ( ms.transform ( feature_matrix [ test ] ) ) == target_array [ test ] )
    )
    # Append logistic regression accuracy
    accuracy_lr.append (
        np.mean ( logistic_regression.predict ( ms.transform ( feature_matrix [ test ] ) ) == target_array [ test ] )
    )

logger.info ( f"Baseline Accuracy: { np.mean ( baseline ) }" )
logger.info ( f"Adaline Accuracy: { np.mean ( accuracy_adaline ) }" )
logger.info ( f"Logistic Regression Accuracy: { np.mean ( accuracy_lr ) }" )


# # Glass Data Set
# ## Extract, Transform, Load: Glass Data
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Glass+Identification
# 
# ### Attribute Information: 10 Attributes (d)
# 
# 1. Id number: 1 to 214 
# 2. RI: refractive index 
# 3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10) 
# 4. Mg: Magnesium 
# 5. Al: Aluminum 
# 6. Si: Silicon 
# 7. K: Potassium 
# 8. Ca: Calcium 
# 9. Ba: Barium 
# 10. Fe: Iron 
# 
# ### One Class Label
# 11. Type of glass: (class attribute) 
#         - 1 building_windows_float_processed
#         - 2 building_windows_non_float_processed
#         - 3 vehicle_windows_float_processed
#         - 4 vehicle_windows_non_float_processed (none in this database) 
#         - 5 containers
#         - 6 tableware
#         - 7 headlamps 

# In[46]:


# Log ETL: Glass Data
logger.info ( "ETL: Glass Data Set" )
# Read Glass Data
glass_data = pd.read_csv ( "glass.data.csv",
    header = None,
    # Assign Column labels
    names = [
        "id_number",
        "refractive_index",
        "sodium",
        "magnesium",
        "aluminum",
        "silicon",
        "potassium",
        "calcium",
        "barium",
        "iron",
        "class",
    ],
)


# In[47]:


# Verify whether any values are null
glass_data.isnull().values.any()


# In[48]:


# Just to be sure
# Replace "?" instance
glass_data.replace ( "?", np.NaN )
# Drop na
glass_data.dropna ( axis = 0, inplace = True )


# In[49]:


# Again
glass_data.isna().any()


# **Notes**
# 
# We see no NaN values in any of the columns

# In[50]:


# Confirm data was properly read by examining data frame
glass_data.info()


# **Notes** 
# 
# As expected, Dataframe shows us we have 214 total observations & 11 columns (or, 11 variables). Each variable has 214 observations, so we know we have no missing data. 
# 
# We see that the following variables are integers: id_number, class
# 
# We see that the following variables are floats: refractive_index, sodium, magnesium, aluminum, silicon, potassium, calcium, barium, iron

# In[51]:


# Look at first few rows of dataframe
glass_data.head()


# **Notes**
# 
# Again, we see we'll need to eliminate id_number.
# 
# We also see that data instances here are continuous (float). We may want to discretize the boundaries for each variable.

# In[52]:


# Drop id_number from cancer data set
glass_data.drop ( [ 'id_number' ], axis = 'columns', inplace = True )
# Confirm
glass_data.head()


# In[53]:


# Log ETL
logger.info ( f"ETL, Dataframe Info: { glass_data.head() }" )


# ## (Brief) Exploratory Data Analysis: Glass Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5

# In[54]:


# Log EDA: Glass Data
logger.info ( "EDA: Glass Data Set" )

# Descriptive Statistics
glass_data.describe()


# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 
# We'll likely want to discretize these attributes by class

# In[55]:


# Log Descriptives
logger.info ( f"Descriptive Statistics: { glass_data.describe() }" )


# ## (Brief) Exploratory Data Analysis: Glass Data
# 
# ### Pair-Wise: Attribute by Class

# In[56]:


# Frequency of glass classifications
glass_data [ 'class' ].value_counts() # raw counts


# **Notes**
# 
# Number of observations for each glass classification:
#     - 2 (building_windows_non_float_processed): 76
#     - 1 (building_windows_float_processed): 70
#     - 7 (headlamps): 29
#     - 3 (vehicle_windows_float_processed): 17
#     - 5 (containers): 13
#     - 6 (tableware): 9

# In[57]:


#Log Pair-Wise Attribute by Class
logger.info ( f"Pair-Wise Attribute by Class: { glass_data [ 'class' ].value_counts() }" )


# In[58]:


# Plot diagnosos frequencies
sns.countplot ( glass_data [ 'class' ],label = "Count" ) # boxplot


# **Notes**
# 
# Number of observations for each glass classification:
#     - 2 (building_windows_non_float_processed): 76
#     - 1 (bhilding_windows_float_processed): 70
#     - 7 (headlamps): 29
#     - 3 (vehicle_windows_float_processed): 17
#     - 5 (containers): 13
#     - 6 (tableware): 9

# In[59]:


# Descriptive Statistics: Describe each variable by class (means only)
glass_data.groupby ( [ 'class' ] )[ 'refractive_index', "sodium", "magnesium", "aluminum","silicon","potassium","calcium","barium","iron" ].mean()


# In[60]:


# Descriptive Statistics: Describe each variable by class (all variables)
glass_data.groupby ( [ 'class' ] ) [ 'refractive_index', "sodium", "magnesium", "aluminum","silicon","potassium","calcium","barium","iron" ].describe()        


# In[61]:


boxplot = glass_data.boxplot ( column = [ 'refractive_index' ], by = [ 'class' ] )


# In[62]:


boxplot = glass_data.boxplot ( column = [ 'sodium' ], by = [ 'class' ] )


# In[63]:


boxplot = glass_data.boxplot ( column = [ "magnesium" ], by = [ 'class' ] )


# In[64]:


boxplot = glass_data.boxplot ( column = [ "aluminum" ], by = [ 'class' ] )


# In[65]:


boxplot = glass_data.boxplot ( column = [ "silicon" ], by = [ 'class' ] )


# In[66]:


boxplot = glass_data.boxplot ( column = [ "potassium" ], by = [ 'class' ] )


# In[67]:


boxplot = glass_data.boxplot ( column = [ "calcium" ], by = [ 'class' ] )


# In[68]:


boxplot = glass_data.boxplot ( column = [ "barium" ], by = [ 'class' ] )


# In[69]:


boxplot = glass_data.boxplot ( column = [ "iron" ], by = [ 'class' ] )


# In[70]:


# Descriptive Statistics: Describe each variable by class
# Refractive Index by Class
describe_by_category ( glass_data, "refractive_index", "class", transpose = True )


# **Notes**
# 
# We observe that for each class, the arithmetic mean of refractive index has very little variability.  We can discretize the boundary to roughly 1.518 for all classes

# In[71]:


# Descriptive Statistics: Describe each variable by class
# Sodium by Class
describe_by_category ( glass_data, "sodium", "class", transpose = True )


# **Notes**
# 
# We see here that values range from 10.73 to 15.79, with mean values hovering around 13.2, 12.8, and 14.5. It would be logical to discretize into: 10.5, 12.5, 14.5

# In[72]:


# Descriptive Statistics: Describe each variable by class
# Magnesium by Class
describe_by_category ( glass_data, "magnesium", "class", transpose = True )


# **Notes**
# 
# We see here that values range from 0 (min class 2) to 4.49 (max class 1). Means range from 3.55 (Class 1), 3.00 (Class 2), 3.54 (Class 3), 0.77 (Class 5), 1.30 (Class 6), and 0.538 (Class 7). We'll discretize to 1, 2, 3, 4

# In[73]:


# Descriptive Statistics: Describe each variable by class
# Aluminum by Class
describe_by_category ( glass_data, "aluminum", "class", transpose = True )


# **Notes**
# 
# We see here that values range from 0.29 (Min Class 1) to 3.5 (Max Class 5). Means range from 1.16 (Class 1), 1.40 (Class 2), 1.2 (Class 3), 2.0 (Class 5),1.36 (Class 6), 2.12 (Class 7) and stds range from 0.27 up to 0.69. We'll; discretize to 0.5, 1, 1.5, 2, 2.5

# In[74]:


# Descriptive Statistics: Describe each variable by class
# Silicon by Class
describe_by_category ( glass_data, "silicon", "class", transpose = True )


# **Notes**
# 
# We see here that values range from 69.81 (Min Class 2) to 75.41 (Max Class 6). Means are all roughly equivalent, rangiong from 72.59 (Class 2) to 73.20 (Class 6). We'll discretize to 72.5, 73.0, 73.5, 74.0

# In[75]:


# Descriptive Statistics: Describe each variable by class
# Potassium by Class
describe_by_category ( glass_data, "potassium", "class", transpose = True )


# **Notes**
# 
# We see here that values range from 0 to 6.21 (Max Class 8). We'll discretize to 0.2, 0.6, 1.2, 1.6

# In[76]:


# Descriptive Statistics: Describe each variable by class
# Calcium by Class
describe_by_category ( glass_data, "calcium", "class", transpose = True )


# **Notes**
# 
# We see that the values range from 5.43 (Min Class 7) to 16.19 (Max Class 2). We'll discretize to 6, 9, 12

# In[77]:


# Descriptive Statistics: Describe each variable by class
# Barium by Class
describe_by_category ( glass_data, "barium", "class", transpose = True )


# **Notes**
# 
# We see that the values range from 0 to 3.15. We'll discretize to 0.5, 1.0, 2.0

# In[78]:


# Descriptive Statistics: Describe each variable by class
# Iron by Class
describe_by_category ( glass_data, "iron", "class", transpose = True )


# **Notes**
# 
# We see that the values range from 0 to 0.51 (Max Class 5). We'll discretize to 0.2 and 0.6

# In[79]:


# Discretize
#discretize_boundries_glass = {
    #"refractive_index": [1.518],
    #"sodium": [10.5, 12.5, 14.5],
    #"magnesium": [1, 2, 3, 4],
    #"aluminum": [0.5, 1.5, 2, 2.5],
    #"silicon": [72.5, 73.0, 73.5, 74.0],
    #"potassium": [0.2, 0.6, 1.2, 1.6],
    #"calcium": [6, 9, 12],
    #"barium": [0.5, 1.0, 2.0],
    #"iron": [0.2, 0.6],
#}


# ## Linear Classification Experiments: Glass Data
# 
# ### Assign Feature Matrix & Target Vector

# In[80]:


# Assign Xs & Ys
# Eliminate id_number
# Set class as category y (target array)
# Set attributes as X (feature matrix)

#X_glass, y_glass = discretize_dataframe (
    #glass_data.drop ( "id_number", axis = 1 ), discretize_boundries_glass
#).pipe(
    #lambda df: (
        #df.drop ( "class", axis = 1 ).values,
        #df [ "class" ].astype ( "category" ).cat.codes.values,
    #)
#)

feature_matrix, target_array = (
        glass_data.drop ( [ "class" ], axis = 1 ).values,
        glass_data [ "class" ].astype ( "category" ).cat.codes,
    )


# ### Run Linear Classification
# ### Adaline & Logistic Regression 
# ### Stratified Cross-Fold Validation

# In[81]:


# Log Experiment: Running Linear Classification Experiment using Adaline & Logistic Regression Algorithms on Glass Data
logger.info ( "Running Glass Linear Classification Experiment: Adaline & Logistic Regression" )

# Use 5 folds for cross-validation
kfold = KFoldStratifiedCV ( number_of_folds = 5 )
accuracy_adaline = []
accuracy_lr = []
baseline = []

# Stratifier KFOLD CV
for train, test in kfold.split ( feature_matrix = feature_matrix, target_array = target_array ):
    # Adaline (Must use MAdaline)
    adaline = MulticlassClassifier (
        model_class = lambda * args: Adaline (
                convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
                fit_intercept = True, # If true, will add column of 1s
                max_iter = 5000, # Stopping criteria based on number of updates
                learning_rate = 0.005, # ETA for gradient update
            ),
            classes = np.unique ( target_array ),
            class_kwargs = { i: {} for i in np.unique ( target_array ) },
    )
             
    # Logistic Regression
    logistic_regression = LogisticRegression (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 15000, # Stopping criteria based on number of updates
        learning_rate = 0.005, # ETA for gradient update
    )

    ms = MaxScaler()

    # Fit the models
    # Adaline
    adaline.fit ( ms.fit_transform ( feature_matrix [ train ] ), target_array [ train ] )
    # Logistic Regression
    logistic_regression.fit ( ms.fit_transform ( feature_matrix [ train ] ), target_array [ train ] )

    # Append results
    baseline.append ( np.mean ( mode ( target_array [ train ] ) == target_array [ test ] ) )
    # Adaline
    accuracy_adaline.append (
            np.mean ( adaline.predict ( ms.transform ( feature_matrix [ test ] ) ) == target_array [ test ] )
    )
    # Logistic Regression
    accuracy_lr.append (
            np.mean ( logistic_regression.predict ( ms.transform ( feature_matrix [ test ] ) ) == target_array [ test ] )
    )

logger.info ( f"Baseline Accuracy: { np.mean ( baseline ) }" )
logger.info ( f"Adaline Accuracy: { np.mean ( accuracy_adaline ) }" )
logger.info ( f"Logistic Regression Accuracy: { np.mean ( accuracy_lr ) }" )


# # Iris Data Set
# ## Extract, Transform, Load: Iris Data
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Iris
# 
# ### Attribute Information: 5 Attributes (d)
# 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 
# ### One Class Label
# 5. Type of Iris (class):
#         - Iris Setosa 
#         - Iris Versicolour
#         - Iris Virginica

# In[82]:


# Log ETL: Iris Data
logger.info ( "ETL: Iris Data Set" )

# Read Iris Data
iris_data = pd.read_csv ( "iris.data.csv",
        header = 0,
        # Assign Column Labels (i.e., variables names)
        names = [
            "sepal_length", 
            "sepal_width", 
            "petal_length", 
            "petal_width", 
            "class"],
)


# In[83]:


# Verify whether any values are null
iris_data.isnull().values.any()


# **Notes**
# 
# We observe no null instances

# In[84]:


# Replace "?" instance
iris_data.replace ( "?", np.NaN )
# Drop na
iris_data.dropna ( axis = 0, inplace = True )


# In[85]:


# Again
iris_data.isna().any()


# **Notes**
# 
# We observe no null instances in any of the attribute columns

# In[86]:


# Confirm data was properly read by examining data frame
iris_data.info()


# **Notes**
# 
# We observe that each attribute is a float

# In[87]:


# Look at first few rows of dataframe
iris_data.head()


# In[88]:


# Log ETL
logger.info ( f"ETL, Dataframe Info: { iris_data.head() }" )


# In[89]:


# Classification for Class Label: data frame for this category
iris_data[ "class" ].astype ( "category" ).cat.codes


# ## (Brief) Exploratory Data Analysis: Iris Data
# 
# ### Single Variable

# In[90]:


# Log EDA: Iris Data
logger.info ( "EDA: Iris Data Set" )

# Descriptive Statistics
iris_data.describe()


# **Notes**
# 
# Total number of observations: 149
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$

# In[91]:


# Log Descriptives
logger.info ( f"Descriptive Statistics: { iris_data.describe() }" )


# ## (Brief) Exploratory Data Analysis: Iris Data
# 
# ### Pair-Wise: Attribute by Class

# In[92]:


# Frequency of diagnoses classifications
iris_data [ 'class' ].value_counts() # raw counts


# **Notes**
# 
# We see there are 50 instances of Iris Virginica, 50 instances of Iris Versicolor, and 50 instances of Iris-setosa

# In[93]:


#Log Pair-Wise Attribute by Class
logger.info ( f"Pair-Wise Attribute by Class: { iris_data [ 'class' ].value_counts() }" )


# In[94]:


# Plot diagnosos frequencies
sns.countplot ( iris_data [ 'class' ],label = "Count" ) # boxplot


# In[95]:


# Descriptive Statistics: Describe each variable by class (means only)
iris_data.groupby ( [ 'class' ] )[ 'sepal_length', 'sepal_width', 'petal_length', 'petal_width' ].mean()


# In[96]:


# Descriptive Statistics: Describe each variable by class (means only)
iris_data.groupby ( [ 'class' ] )[ 'sepal_length', 'sepal_width', 'petal_length', 'petal_width' ].describe()


# In[97]:


boxplot = iris_data.boxplot ( column = [ "sepal_length", "sepal_width"], by = [ 'class' ] )


# In[98]:


boxplot = iris_data.boxplot ( column = [ "petal_length", "petal_width" ], by = [ 'class' ] )


# In[99]:


# Descriptive Statistics: Attribute by Class
# Sepal Length by Class
describe_by_category ( iris_data, "sepal_length", "class", transpose = True )


# **Notes**
# 
# We see that sepal length ranges from 4.3 (Iris-setosa) to 7.9 (Iris-virginica).The mean for Iris-setosa is 5.00 (std 0.355). The mean for Iris-versicolor is 5.9 (std 0.516). The mean for Iris-virginica is 6.5 (std 0.635). We'll discretize using 4.5, 5.5, 6.5, 7.5.

# In[100]:


# Descriptive Statistics: Attribute by Class
# Sepal Width by Class
describe_by_category ( iris_data, "sepal_width", "class", transpose = True )


# **Notes**
# 
# We see that sepal width ranges from 2.0 (Iris-versicolor) to 4.4 (Iris-setosa).The mean for Iris-setosa is 3.41 (std 0.384). The mean for Iris-versicolor is 2.77 (std 0.516). The mean for Iris-virginica is 2.97 (std 0.322). We'll discretize using 2, 3, 4.

# In[101]:


# Descriptive Statistics: Attribute by Class
# Petal Length by Class
describe_by_category ( iris_data, "petal_length", "class", transpose = True )


# **Notes**
# 
# We see that petal length ranges from 1.0 (Iris-versicolor) to 6.9 (Iris-virginica).The mean for Iris-setosa is 1.46 (std 0.175). The mean for Iris-versicolor is 4.26 (std 0.469). The mean for Iris-virginica is 5.55 (std 0.355). We'll discretize using 1, 2, 4, 6.

# In[102]:


# Descriptive Statistics: Attribute by Class
# Petal Width by Class
describe_by_category ( iris_data, "petal_width", "class", transpose = True )


# **Notes**
# 
# We see that petal width ranges from 0.10 (Iris-setosa) to 2.5 (Iris-virginica). The mean for Iris-setosa is 0.244 (std 0.108). The mean for Iris-versicolor is 1.32 (std 0.197). The mean for Iris-virginica is 2.02 (std 0.27). We'll discretize using 0.5, 1, 1.5, 2

# In[103]:


# Discretize the data
#X_iris, y_iris = discretize_dataframe (
    #iris_data,
    #discretize_boundries = {
        #"sepal_length": [4.5, 5.5, 6.5, 7.5],
        #"sepal_width": [2, 3, 4],
        #"petal_width": [0.5, 1, 1.5, 2],
        #"petal_length": [1, 2, 4, 6],
    #},
#).pipe(
    #lambda df: (
        #df.filter ( like = "]" ).values,
        #df [ "class" ].astype ( "category" ).cat.codes.values,
    #)
#)


# ## Linear Classification Experiments: Soybean Data
# 
# ### Assign Feature Matrix & Target Vector
# 

# In[104]:


# Drop columns with no variance
iris_data.pipe ( lambda df: df.loc ( axis = 1 )[ df.nunique() > 1 ] )  


# In[105]:


# Set class as category y (target array)
# Set attributes as X (features matrix)
feature_matrix, target_array = (
    iris_data.drop ( [ "class" ], axis = 1 ).values,
    iris_data [ "class" ].astype ( "category" ).cat.codes,
)


# ### Run Linear Classification
# ### Adaline & Logistic Regression 
# ### Stratified Cross-Fold Validation

# In[106]:


# Experiment: Iris Data

# Log Experiment: Running Linear Classification Experiment using Adaline & Logistic Regression Algorithms on Iris Data
logger.info ( "Running Iris Linear Classification Experiment: Adaline & Logistic Regression" )

# Use 5 folds for cross-validation
kfold = KFoldStratifiedCV ( number_of_folds = 5 )
accuracy_adaline = []
accuracy_lr = []
baseline = []

# Stratified KFold CV
for train, test in kfold.split ( feature_matrix = feature_matrix, target_array = target_array ):
    # Must use MAdaline (Multiple Adaline)
    adaline = MulticlassClassifier (
        model_class = lambda * args: Adaline (
            convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
            fit_intercept = True, # If true, will add column of 1s
            max_iter = 1000, # Stopping criteria based on number of updates
            learning_rate = 0.005, # ETA for gradient update
        ),
        classes = np.unique ( target_array ),
        class_kwargs = { i: {} for i in np.unique ( target_array ) },
    )
    
    # Logistic Regression
    logistic_regression = LogisticRegression (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 1000, # Stopping criteria based on number of updates
        learning_rate = 0.005, # ETA for gradient update
    )

    ms = MaxScaler()

    # Fit models
    # Fit Adaline
    adaline.fit ( ms.fit_transform ( feature_matrix [ train ] ), target_array [ train ] )
    # Fit Logistic Regression
    logistic_regression.fit ( ms.fit_transform ( feature_matrix [ train ] ), target_array [ train ] )
    
    # Save test prediction results to csv file (logistic regression)
    pd.DataFrame (
        np.hstack (
            [
                feature_matrix [ test ], 
                target_array [ test ].values.reshape ( -1, 1 ),
                np.array ( logistic_regression.predict ( ms.transform ( feature_matrix [ test ] ) ) ).reshape ( -1, 1 ),
            ]
        ), columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class', 'prediction' ]
    ).to_csv ( "logistic_regression_iris_predictions.csv",index = False )
    
    # Save test prediction results to csv file (adaline)
    pd.DataFrame (
        np.hstack (
            [
                feature_matrix [ test ], 
                target_array [ test ].values.reshape ( -1, 1 ),
                np.array ( adaline.predict ( ms.transform ( feature_matrix [ test ] ) ) ).reshape ( -1, 1 ),
            ]
        ), columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class', 'prediction' ]
    ).to_csv ( "adaline_iris_predictions.csv", index = False )

    # Append results
    baseline.append ( np.mean ( mode ( target_array [ train ] ) == target_array [ test ] ) )
    # Append adaline accuracy
    accuracy_adaline.append (
        np.mean ( adaline.predict ( ms.transform ( feature_matrix [ test ] ) )  == target_array [ test ] )
    )
    # Append logistic regression accuracy 
    accuracy_lr.append (
        np.mean ( logistic_regression.predict ( ms.transform ( feature_matrix [ test ] ) ) == target_array [ test ])
    )

logger.info ( f"Baseline Accuracy: { np.mean ( baseline ) }" )
logger.info ( f"Adaline Accuracy: { np.mean ( accuracy_adaline ) }" )
logger.info ( f"Logistic Regression Accuracy: { np.mean ( accuracy_lr ) }" )


# # Soybean Data Set
# ## Extract, Transform, Load: Soybean Data
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Soybean+%28Small%29
# 
# ### Attribute Information: 10 Attributes (d)
# 
# 1. date: april,may,june,july,august,september,october,?. 
# 2. plant-stand: normal,lt-normal,?. 
# 3. precip: lt-norm,norm,gt-norm,?. 
# 4. temp: lt-norm,norm,gt-norm,?. 
# 5. hail: yes,no,?. 
# 6. crop-hist: diff-lst-year,same-lst-yr,same-lst-two-yrs, same-lst-sev-yrs,?. 
# 7. area-damaged: scattered,low-areas,upper-areas,whole-field,?. 
# 8. severity: minor,pot-severe,severe,?. 
# 9. seed-tmt: none,fungicide,other,?. 
# 10. germination: 90-100%,80-89%,lt-80%,?. 
# 11. plant-growth: norm,abnorm,?. 
# 12. leaves: norm,abnorm. 
# 13. leafspots-halo: absent,yellow-halos,no-yellow-halos,?. 
# 14. leafspots-marg: w-s-marg,no-w-s-marg,dna,?. 
# 15. leafspot-size: lt-1/8,gt-1/8,dna,?. 
# 16. leaf-shread: absent,present,?. 
# 17. leaf-malf: absent,present,?. 
# 18. leaf-mild: absent,upper-surf,lower-surf,?. 
# 19. stem: norm,abnorm,?. 
# 20. lodging: yes,no,?. 
# 21. stem-cankers: absent,below-soil,above-soil,above-sec-nde,?. 
# 22. canker-lesion: dna,brown,dk-brown-blk,tan,?. 
# 23. fruiting-bodies: absent,present,?. 
# 24. external decay: absent,firm-and-dry,watery,?. 
# 25. mycelium: absent,present,?. 
# 26. int-discolor: none,brown,black,?. 
# 27. sclerotia: absent,present,?. 
# 28. fruit-pods: norm,diseased,few-present,dna,?. 
# 29. fruit spots: absent,colored,brown-w/blk-specks,distort,dna,?. 
# 30. seed: norm,abnorm,?. 
# 31. mold-growth: absent,present,?. 
# 32. seed-discolor: absent,present,?. 
# 33. seed-size: norm,lt-norm,?. 
# 34. shriveling: absent,present,?. 
# 35. roots: norm,rotted,galls-cysts,?.
# 
# ### One Class Label
# 
# 36. Class

# In[107]:


# Log ETL: Soybean Data
logger.info ( "ETL: Soybean Data Set" )

# Read Soybean Data
soybean_data = pd.read_csv ( "soybean-small.data.csv",
        header = 0,
        # Assign Column Labels (i.e., variables names)
        names = [
            "date",
            "plant-stand",
            "precip",
            "temp",
            "hail",
            "crop-hist",
            "area-damaged",
            "severity",
            "seed-tmt",
            "germination",
            "plant-growth",
            "leaves",
            "leafspots-halo",
            "leafspots-marg",
            "leafspot-size",
            "leaf-shread",
            "leaf-malf",
            "leaf-mild",
            "stem",
            "lodging",
            "stem-cankers",
            "canker-lesion",
            "fruiting-bodies",
            "external decay",
            "mycelium",
            "int-discolor",
            "sclerotia",
            "fruit-pods",
            "fruit spots",
            "seed",
            "mold-growth",
            "seed-discolor",
            "seed-size",
            "shriveling",
            "roots",
            "instance_class",
        ],
    )


# In[108]:


# Verify whether any values are null
soybean_data.isnull().values.any()


# **Note**
# 
# We see there are no null instances

# In[109]:


# Replace "?" instance
soybean_data.replace ( "?", np.NaN )
# Drop na
soybean_data.dropna ( axis = 0, inplace = True )


# In[110]:


# Again
soybean_data.isna().any()


# **Notes**
# 
# Again, we find no NaN instances in any of the column attributes. We also observe that the last column is the class label

# In[111]:


# Confirm data was properly read by examining data frame
soybean_data.info()


# **Notes**
# 
# We see n = 46 and there are 46 instances in each attribute column (thus, no missing data). All attribute (Xs) columns are integers and the instance class is an object.

# ## (Brief) Exploratory Data Analysis: Soybean Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5

# In[112]:


# Look at first few rows
soybean_data.head()


# **Notes**
# 
# We see that we'll need to use one-hot encoding for the instance class and we may need to eliminate columns where data has no variance. These will not help the algorithm learn classes. 

# In[113]:


# Drop columns with no variance
for col in soybean_data.columns:
    if len ( soybean_data [ col ].unique()) == 1:
        soybean_data.drop ( col,inplace = True, axis = 1 )
# Confirm
soybean_data.head()


# In[114]:


print ( soybean_data.columns )


# In[115]:


# Rename column
soybean_data.rename ( columns = { "instance_class":"class" }, inplace = True)


# In[116]:


# Assign class as category label
soybean_data [ "class" ].astype ( "category" ).cat.codes
# Confirm
soybean_data.head()


# In[117]:


# Log ETL
logger.info ( f"ETL, Dataframe Info: { soybean_data.head() }" )


# In[118]:


# Find Y Categories
print ( soybean_data [ "class" ] )


# In[119]:


# Map class values to values
# D1 = 0; D2 = 1; D3 = 2; D4 = 3
soybean_data [ 'class' ] = soybean_data [ 'class' ].map ( { 'D1' : 0, 'D2' : 1, 'D3' : 2, 'D4': 3 } )
# Confirm
soybean_data.head()


# In[120]:


print ( soybean_data [ 'class' ] )


# In[121]:


# One-hot encoding/Dummy variables for all attribute columns
soybean_bool = pd.get_dummies (
    soybean_data,
    columns = [ col for col in soybean_data.columns if col != "class" ],
    drop_first = True,
)

# Confirm
soybean_bool.head()


# In[122]:


# Log ETL
logger.info ( f"ETL, Dataframe Info, One-Hot Encoding: { soybean_bool.head() }" )


# ## (Brief) Exploratory Data Analysis: Soybean Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5
# 

# In[123]:


# Log EDA: Soybean Data
logger.info ( "EDA: Soybean Data Set" )

# Descriptive Statistics
soybean_data.describe()


# **Notes**
# 
# Total number of observations: 46
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 

# In[124]:


# Log Descriptives
logger.info ( f"Descriptive Statistics: { soybean_data.describe() }" )


# ## (Brief) Exploratory Data Analysis: Soybean Data
# 
# ### Pair-Wise: Attribute by Class

# In[125]:


# Frequency of diagnoses classifications
soybean_data [ 'class' ].value_counts() # raw counts


# **Notes**
# 
# Here we see 9 observations in class 0, 10 observations in class 1, 10 observations in class 2, and 17 observations in class 3

# In[126]:


#Log Pair-Wise Attribute by Class
logger.info ( f"Pair-Wise Attribute by Class: { soybean_data [ 'class' ].value_counts() }" )


# In[127]:


# Plot diagnosos frequencies
sns.countplot ( soybean_data [ 'class' ],label = "Count" ) # boxplot


# In[128]:


# Get Columns again
list ( soybean_data.columns )


# In[129]:


# Descriptive Statistics: Describe each variable by class (means only)
soybean_data.groupby ( [ 'class' ] )[ 'date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt','germination','leaves','lodging','stem-cankers','canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods','roots' ].mean()


# In[130]:


# Descriptive Statistics: Describe each variable by class (means only)
soybean_data.groupby ( [ 'class' ] )[ 'date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt','germination','leaves','lodging','stem-cankers','canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods','roots' ].describe()


# In[131]:


boxplot = soybean_data.boxplot ( column = [ "date", "plant-stand"], by = [ 'class' ] )


# In[132]:


boxplot = soybean_data.boxplot ( column = [ 'precip','temp' ], by = [ 'class' ] )


# In[133]:


boxplot = soybean_data.boxplot ( column = [ 'hail','crop-hist' ], by = [ 'class' ] )


# In[134]:


boxplot = soybean_data.boxplot ( column = [ 'area-damaged','severity' ], by = [ 'class' ] )


# In[135]:


boxplot = soybean_data.boxplot ( column = [ 'seed-tmt','germination' ], by = [ 'class' ] )


# In[136]:


boxplot = soybean_data.boxplot ( column = [ 'leaves','lodging' ], by = [ 'class' ] )


# In[137]:


boxplot = soybean_data.boxplot ( column = [ 'stem-cankers','canker-lesion' ], by = [ 'class' ] )


# In[138]:


boxplot = soybean_data.boxplot ( column = [ 'fruiting-bodies', 'external decay' ], by = [ 'class' ] )


# In[139]:


boxplot = soybean_data.boxplot ( column = [ 'mycelium', 'int-discolor' ], by = [ 'class' ] )


# In[140]:


boxplot = soybean_data.boxplot ( column = [ 'sclerotia', 'fruit-pods','roots' ], by = [ 'class' ] )


# In[141]:


# Descriptive Statistics: Attribute by Class
# Date by Class
describe_by_category ( soybean_data, "date", "class", transpose = True )


# In[142]:


# Descriptive Statistics: Attribute by Class
# Plant-Stand by Class
describe_by_category ( soybean_data, "plant-stand", "class", transpose = True )


# In[143]:


# Descriptive Statistics: Attribute by Class
# precip by Class
describe_by_category ( soybean_data, "precip", "class", transpose = True )


# In[144]:


# Descriptive Statistics: Attribute by Class
# Temp by Class
describe_by_category ( soybean_data, "temp", "class", transpose = True )


# In[145]:


# Descriptive Statistics: Attribute by Class
# Hail by Class
describe_by_category ( soybean_data, "hail", "class", transpose = True )


# In[146]:


# Descriptive Statistics: Attribute by Class
# Crop-Hist by Class
describe_by_category ( soybean_data, "crop-hist", "class", transpose = True )


# In[147]:


# Descriptive Statistics: Attribute by Class
# Area-Damaged by Class
describe_by_category ( soybean_data, "area-damaged", "class", transpose = True )


# In[148]:


# Descriptive Statistics: Attribute by Class
# Severity by Class
describe_by_category ( soybean_data, "severity", "class", transpose = True )


# In[149]:


# Descriptive Statistics: Attribute by Class
# Seed-tmt by Class
describe_by_category ( soybean_data, "seed-tmt", "class", transpose = True )


# In[150]:


# Descriptive Statistics: Attribute by Class
# Germination by Class
describe_by_category ( soybean_data, "germination", "class", transpose = True )


# In[151]:


# Descriptive Statistics: Attribute by Class
# Leaves by Class
describe_by_category ( soybean_data, "leaves", "class", transpose = True )


# In[152]:


# Descriptive Statistics: Attribute by Class
# Lodging by Class
describe_by_category ( soybean_data, "lodging", "class", transpose = True )


# In[153]:


# Descriptive Statistics: Attribute by Class
# Stem-Cankers by Class
describe_by_category ( soybean_data, "stem-cankers", "class", transpose = True )


# In[154]:


# Descriptive Statistics: Attribute by Class
# Canker-lesion by Class
describe_by_category ( soybean_data, "canker-lesion", "class", transpose = True )


# In[155]:


# Descriptive Statistics: Attribute by Class
# Fruiting-bodies by Class
describe_by_category ( soybean_data, "fruiting-bodies", "class", transpose = True )


# In[156]:


# Descriptive Statistics: Attribute by Class
# External decay by Class
describe_by_category ( soybean_data, "external decay", "class", transpose = True )


# In[157]:


# Descriptive Statistics: Attribute by Class
# Mycelium by Class
describe_by_category ( soybean_data, "mycelium", "class", transpose = True )


# In[158]:


# Descriptive Statistics: Attribute by Class
# int-discolor by Class
describe_by_category ( soybean_data, "int-discolor", "class", transpose = True )


# In[159]:


# Descriptive Statistics: Attribute by Class
# Sclerotia by Class
describe_by_category ( soybean_data, "sclerotia", "class", transpose = True )


# In[160]:


# Descriptive Statistics: Attribute by Class
# Fruit-Pods by Class
describe_by_category ( soybean_data, "fruit-pods", "class", transpose = True )


# In[161]:


# Descriptive Statistics: Attribute by Class
# Roots by Class
describe_by_category ( soybean_data, "roots", "class", transpose = True )


# ## Linear Classification Experiments: Soybean Data
# 
# ### Assign Feature Matrix & Target Vector
# 

# In[162]:


# X = Feature Matrix/Input Matrix
# y = target array/ class label
feature_matrix, target_array = (
    pd.get_dummies(
        soybean_bool.drop ( "class", axis = 1 ),
        columns = soybean_bool.drop ( "class", axis = 1 ).columns,
        drop_first = True,
    ).values,
    soybean_bool [ "class" ].values,
)


# ### Run Linear Classification
# ### Adaline & Logistic Regression 
# ### Stratified Cross-Fold Validation

# In[163]:


# Experiment: Soybean Data

# Log Experiment: Running Linear Classification Experiment using Adaline & Logistic Regression Algorithms on Soybean Data
logger.info ( "Running Soybean Linear Classification Experiment: Adaline & Logistic Regression" )

# Use 5 folds for cross-validation
kfold = KFoldStratifiedCV ( number_of_folds = 5 )
accuracy_adaline = []
accuracy_lr = []
baseline = []

# Stratified KFold CV
for train, test in kfold.split ( feature_matrix = feature_matrix, target_array = target_array ):
    # MAdaline (Multiple Adaline)
    adaline = MulticlassClassifier (
        model_class = lambda * args: Adaline (
            convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
            fit_intercept = True, # If true, will add column of 1s
            max_iter = 1000, # Stopping criteria based on number of updates
            learning_rate = 0.001, # ETA for gradient update
        ),
        classes = np.unique ( target_array ),
        class_kwargs = { i: {} for i in np.unique ( target_array ) },
    )
    
    # Logistic Regression
    logistic_regression = LogisticRegression (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 1000, # Stopping criteria based on number of updates
        learning_rate = 0.001, # ETA for gradient update
    )

    # Fit the models on the transformed training data
    # Fit Adaline
    adaline.fit ( feature_matrix [ train ], target_array [ train ] )

    # Fit Logistic Regression
    logistic_regression.fit ( feature_matrix [ train ], target_array [ train ] )

    # Append accuracy
    baseline.append ( np.mean ( mode ( target_array [ train ] ) == target_array [ test ] ) )
    # Append adaline accuracy
    accuracy_adaline.append ( np.mean ( adaline.predict ( feature_matrix [ test ] ) == target_array [ test ] ) )
    # Append logistic regression accuracy 
    accuracy_lr.append ( np.mean ( logistic_regression.predict ( feature_matrix [ test ] ) == target_array [ test ] ))

logger.info ( f"Baseline Accuracy: { np.mean ( baseline ) }" )
logger.info ( f"Adaline Accuracy: { np.mean ( accuracy_adaline ) }" )
logger.info ( f"Logistic Regression Accuracy: { np.mean ( accuracy_lr ) }" )


# # House Votes Data Set
# ## Extract, Transform, Load: House Votes Data
# 
# Data obtained from https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
# 
# ### Attribute Information: 17 Attributes (d)
# 
# 1. Class Name: 2 (democrat, republican) 
# 2. handicapped-infants: 2 (y,n) 
# 3. water-project-cost-sharing: 2 (y,n) 
# 4. adoption-of-the-budget-resolution: 2 (y,n) 
# 5. physician-fee-freeze: 2 (y,n) 
# 6. el-salvador-aid: 2 (y,n) 
# 7. religious-groups-in-schools: 2 (y,n) 
# 8. anti-satellite-test-ban: 2 (y,n) 
# 9. aid-to-nicaraguan-contras: 2 (y,n) 
# 10. mx-missile: 2 (y,n) 
# 11. immigration: 2 (y,n) 
# 12. synfuels-corporation-cutback: 2 (y,n) 
# 13. education-spending: 2 (y,n) 
# 14. superfund-right-to-sue: 2 (y,n) 
# 15. crime: 2 (y,n) 
# 16. duty-free-exports: 2 (y,n) 
# 17. export-administration-act-south-africa: 2 (y,n)
# 
# ### One Class Label
# 18. Instance Class

# In[164]:


# Log ETL: House Vote Data
logger.info ( "ETL: Vote Data Set" )

# Read House Vote Data
house_votes_data = (
    pd.read_csv(
        "house-votes-84.data.csv",
        header = None,
        # Assign column labels
        names = [
            "class",
            "handicapped-infants",
            "water-project-cost-sharing",
            "adoption-of-the-budget-resolution",
            "physician-fee-freeze",
            "el-salvador-aid",
            "religious-groups-in-schools",
            "anti-satellite-test-ban",
            "aid-to-nicaraguan-contras",
            "mx-missile",
            "immigration",
            "synfuels-corporation-cutback",
            "education-spending",
            "superfund-right-to-sue",
            "crime",
            "duty-free-exports",
            "export-administration-act-south-africa",
        ],
    )
    .replace ( "?", np.NaN )
    .dropna ( axis = 0, how = 'any')
    .replace ( "y", 1 )
    .replace ( "n", 0 )
)


# In[165]:


# Confirm column names
list ( house_votes_data.columns )


# In[166]:


# Verify whether any values are null
house_votes_data.isnull().values.any()


# In[167]:


# Again
house_votes_data.isna().any()


# **Notes**
# 
# We see no NaN instances in any of the attributes or class columns

# In[168]:


# Assign class as category
house_votes_data[ "class" ].astype ( "category" ).cat.codes


# In[169]:


# Confirm data was properly read by examining data frame
house_votes_data.info()


# **Notes**
# 
# We see there are 17 columns (16 attributes; 1 class), 232 entries total and each column has 232 instances (so no missing data). We see that all columns are listed as objects.

# In[170]:


# Look at first few rows of dataframe
house_votes_data.head()


# In[171]:


# Map 'republican': 1, 'democrat': 0
house_votes_data.replace ( [ 'republican', 'democrat' ], [ 1, 0 ], inplace = True )
house_votes_data.head()


# In[172]:


# Log ETL
logger.info ( f"ETL, Dataframe Info: { house_votes_data.head() }" )


# In[173]:


# Make all instances at integer type
house_votes_data.astype ( "int" )


# In[174]:


# Confirm
house_votes_data.info()


# In[175]:


# One-hot encoding
house_votes_bool = pd.get_dummies (
    house_votes_data,
    dummy_na = True,
    drop_first = True,
    columns = list ( filter ( lambda c: c != "class", house_votes_data.columns ) ),
)


# In[176]:


# Confirm
house_votes_bool.head()


# In[177]:


# Log ETL
logger.info ( f"ETL, Dataframe Info, One-Hot Encoding: { house_votes_bool.head() }" )


# ## (Brief) Exploratory Data Analysis: House Votes Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5

# In[178]:


# Log EDA: House Votes Data
logger.info ( "EDA: House Votes Data Set" )

# Descriptive Statistics
house_votes_data.describe()


# **Notes**
# 
# Total number of observations: 232
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 

# In[179]:


# Log Descriptives
logger.info ( f"Descriptive Statistics: { house_votes_data.describe() }" )


# ## (Brief) Exploratory Data Analysis: House Votes Data
# 
# ### Pair-Wise: Attribute by Class

# In[180]:


# Frequency of diagnoses classifications
house_votes_data [ 'class' ].value_counts() # raw counts


# **Notes*
# 
# We observe 124 Republicans and 108 Democrats

# In[181]:


#Log Pair-Wise Attribute by Class
logger.info ( f"Pair-Wise Attribute by Class: { house_votes_data [ 'class' ].value_counts() }" )


# In[182]:


# Plot diagnosos frequencies
sns.countplot ( house_votes_data [ 'class' ],label = "Count" ) # boxplot


# In[183]:


# Descriptive Statistics: Count by Class
house_votes_data.groupby ( [ 'class' ] )[ "handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-aid","religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa",].count()


# ## Linear Classification Experiments: House Votes Data
# 
# ### Assign Feature Matrix & Target Vector
# 

# In[184]:


# X = feature matrix; input matrix; attributes
# y = target array; class label

feature_matrix, target_array = (
    house_votes_bool.drop ( [ "class" ], axis = 1).values,
    house_votes_bool [ "class" ].values,
)


# ### Run Linear Classification
# ### Adaline & Logistic Regression 
# ### Stratified Cross-Fold Validation

# In[185]:


# Experiment: House Votes Data

# Log Experiment: Running Linear Classification Experiment using Adaline & Logistic Regression Algorithms on House Votes Data
logger.info ( "Running House Votes Linear Classification Experiment: Adaline & Logistic Regression" )

# Use 5 folds for cross-validation
kfold = KFoldStratifiedCV ( number_of_folds = 5 )
accuracy_adaline = []
accuracy_lr = []
baseline = []

# Stratified KFOLD CV
for train, test in kfold.split ( feature_matrix = feature_matrix, target_array = target_array ):
    # Adaline
    adaline = Adaline (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 1000, # Stopping criteria based on number of updates
        learning_rate = 0.0001, # ETA for gradient update
    )
    
    # Logistic Regression
    logistic_regression = LogisticRegression (
        convergence_tolerance = 0.0001, # Stopping criteria based on value of gradient
        fit_intercept = True, # If true, will add column of 1s
        max_iter = 1000, # Stopping criteria based on number of updates
        learning_rate = 0.0001, # ETA for gradient update
    )

    # Fit the models
    # Fit Adaline
    adaline.fit ( feature_matrix [ train ], target_array [ train ] )
    # Fit Logistic Regression
    logistic_regression.fit ( feature_matrix [ train ], target_array [ train ] )

    # Append the results
    baseline.append ( np.mean ( mode ( target_array [ train ] ) == target_array [ test ] ) )
    # Append Adaline accuracy
    accuracy_adaline.append ( np.mean ( adaline.predict ( feature_matrix [ test ] ) == target_array [ test ] ) )
    # Append Logistic Regression accuracy
    accuracy_lr.append ( np.mean ( logistic_regression.predict ( feature_matrix [ test ] ) == target_array [ test ] ) )

logger.info ( f"Baseline Accuracy: { np.mean ( baseline ) }" )
logger.info ( f"Adaline Accuracy: { np.mean ( accuracy_adaline ) }" )
logger.info ( f"Logistic Regression Accuracy: { np.mean ( accuracy_lr ) }" )


# In[ ]:





# In[ ]:




