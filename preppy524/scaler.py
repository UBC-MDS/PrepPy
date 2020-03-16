import pandas as pd

from sklearn.preprocessing import StandardScaler


def scaler(x_train, x_validation, x_test, colnames):
    """
    Perform standard scaler on numerical features.
    Parameters
    ----------
    x_train : pandas.core.frame.DataFrame, numpy array or list
    Dataframe of train set containing columns to be scaled.
    x_validation : pandas.core.frame.DataFrame, numpy array or list
    Dataframe of validation set containing columns to be scaled.
    x_test : pandas.core.frame.DataFrame, numpy array or list
    Dataframe of test set containing columns to be scaled.
    num_columns : list
    A list of numeric column names
    Returns
    -------
    dict
      Stores the scaled and transformed x_train and x_test sets separately as
      two dataframes.
    Examples
    --------
    >>> from PrepPy import prepPy as pp
    >>> x_train = pd.DataFrame(np.array([['Blue', 56, 4], ['Red', 35, 6],
    ['Green', 18, 9]]),
                             columns=['color', 'count', 'usage'])
    >>> x_test = pd.DataFrame(np.array([['Blue', 66, 6], ['Red', 42, 8],
    ['Green', 96, 0]]),
                             columns=['color', 'count', 'usage'])
    >>> x_validation = pd.DataFrame(np.array([['Blue', 30, 18], ['Red', 47, 2],
     ['Green', 100, 4]]),
                             columns=['color', 'count', 'usage'])
    >>> colnames = ['count', 'usage']
    >>> x_train = pp.scaler(x_train, x_validation, x_test, colnames)['x_train']
    >>> x_train
    color   count   usage
    0 Blue 1.26538 -1.13555
    1 Red -0.0857887 -0.162221
    2 Green -1.17959 1.29777

    >>> x_validation = pp.scaler(x_train, x_validation,
            x_test, colnames)['x_validation']
    >>> x_validation
        color  count       usage
    0  Blue    1.80879917 -0.16222142
    1  Red     0.16460209  1.81110711
    2  Green   2.43904552 -4.082207
    >>> x_test = pp.scaler(x_train, x_validation, x_test, colnames)['x_test']
    >>> x_test
       color   count      usage
    0  Blue    1.90879917 -0.16222142
    1  Red     0.36460209  0.81110711
    2  Green   3.83904552 -3.082207
    """
    # Type error exceptions
    if not isinstance(x_train, pd.DataFrame) or \
            not isinstance(x_test, pd.DataFrame) \
            or not isinstance(x_validation, pd.DataFrame):
        raise TypeError('A wrong data type has been passed. Please pass a ' +
                        'dataframe')
    if not isinstance(colnames, list):
        raise TypeError('Numeric column names is not in a list format')
    if ((x_train.empty is True) or (x_test.empty is True) or
            (x_validation.empty is True) or
            (len(colnames) == 0)):
        raise ValueError('Input data cannot be empty')
    scaled_data = {}
    sc = StandardScaler()
    x_train_scaled = x_train.copy()
    x_train_scaled[colnames] = sc.fit_transform(x_train[colnames])
    scaled_data['x_train'] = x_train_scaled
    x_validation_scaled = x_validation.copy()
    x_validation_scaled[colnames] = sc.fit_transform(x_validation[colnames])
    scaled_data['x_validation'] = x_validation_scaled
    x_test_scaled = x_test.copy()
    x_test_scaled[colnames] = sc.fit_transform(x_test[colnames])
    scaled_data['x_test'] = x_test_scaled
    return scaled_data
