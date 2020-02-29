## PrepPy 

![](https://github.com/UBC-MDS/PrepPy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/PrepPy/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/PrepPy) ![Release](https://github.com/UBC-MDS/PrepPy/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/PrepPy/badge/?version=latest)](https://PrepPy.readthedocs.io/en/latest/?badge=latest)

This is a Python package that preprocesses data as follows:

- Identifies features of different data types in a dataframe

- Splits data into train, validation, and test sets

- One-hot encodes features of categorical type

- Performs standard scaling of categorical features

- The package returns preprocessed and split train, validation, and test data sets ready for analysis/modelling

### Installation:

```
pip install -i https://test.pypi.org/simple/ PrepPy
```

### Features
This package has the following features:

- split data set into train, validation, and test sets

- identify data types for each column/feature 

- perform one-hot encoding on the categorical features

- perform standardcscaling on the numerical features

- concat the generated columns to original dataframe

- rename columns



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


### Documentation
The official documentation is hosted on Read the Docs: <https://PrepPy.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
