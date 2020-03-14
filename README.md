## PrepPy 

![](https://github.com/UBC-MDS/PrepPy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/PrepPy/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/PrepPy) ![Release](https://github.com/UBC-MDS/PrepPy/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/preppy524/badge/?version=latest)](https://preppy524.readthedocs.io/en/latest/?badge=latest)

### Package Summary

`PrepPy` is a package for Python to help preprocessing in machine learning tasks. 
There are certain repetitive tasks  that come up often when doing a machine learning project and this package aims to alleviate those chores.
Some of the issues that come up regularly are: finding the types of each column in a dataframe, splitting the data (whether into train/test sets or train/test/validation sets, one-hot encoding,  and scaling features.
This package will help with all of those tasks.

### Installation:

```
pip install -i https://test.pypi.org/simple/ PrepPy
```

### Features

This package has the following features:

- `train_valid_test_split`: This function splits the data set into train, validation, and test sets.

- `data_type`: This function identifies data types for each column/feature. It returns one dataframe for each type of data.

- `one-hot`: This function performs one-hot encoding on the categorical features and returns a dataframe for the train, test, validation sets with sensible column names.

- `scaler`: This function performs standard scaling on the numerical features. 



### Dependencies

- import pandas as pd

- import numpy as np

- from sklearn.preprocessing import OneHotEncoder

- from sklearn.preprocessing import StandardScaler, MinMaxScaler

- from sklearn.model_selection import train_test_split


### Usage

from PrepPy import PrepPy as pp

**Identify features of different data types**
`pp.data_type(my_data)['num']`

`pp.data_type(my_data)['cat']`


**One-hot encode features of categorical type**

`pp.one_hot(my_data)`

**Train, validation, and test split**

`pp.split(my_data)`

**Standard Scaling of categorical features**

`X_train = pp.scaler(x_train, x_test, colnames)['x_train']`

`X_test = pp.scaler(x_train, x_test, colnames)['x_test']`

### Our package in the Python ecosystem

Many of the functions in this package can also be done using the various functions of `sklearn`.
However, some of the functions in `sklearn` take multiple steps to complete what our package can do in one line.
For example, if one wants to split a dataset into train, test, and validation sets, they would need to use `sklearn`'s `train_test_split` twice.
This package's `train_test_val_split` allows users to do this more efficiently.
Further, the one-hot encoder in `sklearn` does not make sensible column names unless the user does some wrangling.
The `one-hot` function in this package will implement `sklearn`'s one-hot encoder, but will wrangle the columns and name them automatically.
Overall, this package fits in well with the Python ecosystem and can help make machine learning a little easier. 


### Documentation
The official documentation is hosted on Read the Docs: <https://PrepPy.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
