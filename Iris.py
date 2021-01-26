#!/usr/bin/python3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.plotting import scatter_matrix
import sys
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}\n\n".format(iris_dataset.keys()))

# print(iris_dataset['DESCR']) ## DESCRICAO DO ARQUIVO

# Target names for the iris species
print("Target names: \n{}\n\n".format(iris_dataset['target_names']))

# Description of each feature
print("Feature names: \n{}\n\n".format(iris_dataset['feature_names']))

# Numeric measurements of the petals and sepals within NumPy array.
print("Type of data: \n{}\n\n".format(type(iris_dataset['data'])))

# The rows inn the data array correspond to flowers, while the columns represent the measurement of each flower.
# It contains measurement for 150 different flowers.
print("Shape of data: \n{}\n\n".format(iris_dataset['data'].shape))

# Here are the first five columns of data.
# From left to right, the matrix represents the sepal length, sepal width, petal length and petal width;
# All in centimeters.
# This array is also a NumPy Array
print("First five columns of data: \n{}\n\n".format(iris_dataset['data'][:5]))

# The species are encoded as integers from 0 to 2:
# 0 means setosa, 1 means versicolor, and 2 means virginica.
print("Target: \n{}\n\n".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# The output of the train_test_split function is X_train, X_test, y_train, and y_test,
# which are all NumPy arrays. X_train contains 75% of the rows of the dataset, and X_test contains the remaining 25%.
print("X_train shape: \n{}\n\n".format(X_train.shape))
print("y_trains shape: \n{}\n\n".format(y_train.shape))

print("X_test shape: \n{}\n\n".format(X_test.shape))
print("y_test shape: \n{}\n\n".format(y_test.shape))

# Create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='0',
    hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
