#!/bin/python3

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}\n\n".format(iris_dataset.keys()))

# print(iris_dataset['DESCR']) ## DESCRICAO DO ARQUIVO

# Target names for the iris species
print("Target names: \n{}\n\n".format(iris_dataset['target_names']))

# Description of each feature
print("Feature names: \n{}\n\n".format(iris_dataset['feature_names']))

# Numeric measurements of the petals and sepals within numpy array.
print("Type of data: \n{}\n\n".format(type(iris_dataset['data'])))

# The rows inn the data array correspond to flowers, while the columns represent the measurement of each flower.
# It contains measurement for 150 different flowers.
print("Shape of data: \n{}\n\n".format(iris_dataset['data'].shape))

