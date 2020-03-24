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
pip install -i https://test.pypi.org/simple/ preppy524
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

#### PrepPy.datatype module
The `data_type()` function identifies features of different data types: numeric or categorical.  

| <span style="font-weight:normal">Input</span>  | <span style="font-weight:normal">Pandas DataFrame</span>    |
|---|----------------|
| Output |A tuple (Pandas DataFrame of numeric features, Pandas DataFrame of categorical features)| 

```
from preppy524 import datatype  
datatype.data_type(my_data)
```

**Example: **  
```
my_data = pd.DataFrame({'fruits': ['apple', 'banana', 'pear'],
                        'count': [3, 5, 8],
                        'price': [1.0, 6.5, 9.23]})
```

`datatype.data_type(my_data)[0]`

|  |count| price |
|---|----|----|
| 0 |     3 |   1.0 |
| 1 |     5 |   6.5 |
| 2 |     8 |  9.23 |

`datatype.data_type(my_data)[1]`

|  | fruits |
|---|--------|
| 0 | apple |
| 1 | banana |
| 2 | pear |

### Our package in the Python ecosystem

Many of the functions in this package can also be done using the various functions of `sklearn`.
However, some of the functions in `sklearn` take multiple steps to complete what our package can do in one line.
For example, if one wants to split a dataset into train, test, and validation sets, they would need to use `sklearn`'s `train_test_split` twice.
This package's `train_test_val_split` allows users to do this more efficiently.
Further, the one-hot encoder in `sklearn` does not make sensible column names unless the user does some wrangling.
The `one-hot` function in this package will implement `sklearn`'s one-hot encoder, but will wrangle the columns and name them automatically.
Overall, this package fits in well with the Python ecosystem and can help make machine learning a little easier. 


### Documentation
The official documentation is hosted on Read the Docs: <https://preppy524.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
