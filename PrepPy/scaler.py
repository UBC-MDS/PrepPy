def scaler(X_train, X_validation, X_test, colnames):
    """
    Perform standard scaler on numerical features. 
    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame, numpy array or list
    Dataframe of train set containing columns to be scaled.
    X_validation : pandas.core.frame.DataFrame, numpy array or list
    Dataframe of validation set containing columns to be scaled.
    X_test : pandas.core.frame.DataFrame, numpy array or list
    Dataframe of test set containing columns to be scaled.  
    num_columns : list
    A list of numeric column names
    Returns
    -------
    dict
      Stores the scaled and transformed X_train and X_test sets separately as two dataframes.
    Examples
    --------
    >>> from PrepPy import prepPy as pp
    >>> x_train = pd.DataFrame(np.array([['Blue', 56, 4], ['Red', 35, 6], ['Green', 18, 9]]),
                             columns=['color', 'count', 'usage'])
    >>> x_test = pd.DataFrame(np.array([['Blue', 66, 6], ['Red', 42, 8], ['Green', 96, 0]]),
                             columns=['color', 'count', 'usage'])
    >>> X_validation = pd.DataFrame(np.array([['Blue', 30, 18], ['Red', 47, 2], ['Green', 100, 4]]),
                             columns=['color', 'count', 'usage'])
    >>> colnames = ['count', 'usage']                          
    >>> X_train = pp.scaler(x_train, X_validation, x_test, colnames)['x_train']
    >>> X_train    
    color   count   usage
    0 Blue 1.26538 -1.13555
    1 Red -0.0857887 -0.162221
    2 Green -1.17959 1.29777
    >>> X_validation = pp.scaler(x_train, X_validation, x_test, colnames)['X_validation']
    >>> X_validation
        color  count       usage
    0  Blue    1.80879917 -0.16222142
    1  Red     0.16460209  1.81110711
    2  Green   2.43904552 -4.082207 
    >>> X_test = pp.scaler(x_train, X_validation, x_test, colnames)['x_test']
    >>> X_test
       color   count      usage
    0  Blue    1.90879917 -0.16222142
    1  Red     0.36460209  0.81110711
    2  Green   3.83904552 -3.082207 
    """

    
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd  
    
    #Type error exceptions
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame) or not isinstance(X_validation, pd.DataFrame):
        raise TypeError('A wrong data type has been passed. Please pass a dataframe')

    if not isinstance(colnames, list):
        raise TypeError('Numeric column names is not in a list format')
    
    if ((X_train.empty == True) or (X_test.empty == True) or (X_validation.empty == True) or (len(colnames) == 0)):
        raise ValueError('Input data cannot be empty')
    
    scaled_data = {}

    sc = StandardScaler()

    
    X_train_scaled = X_train.copy()
    X_train_scaled[colnames] = sc.fit_transform(X_train[colnames])
    scaled_data['X_train'] = X_train_scaled
    
    X_validation_scaled = X_validation.copy()
    X_validation_scaled[colnames] = sc.fit_transform(X_validation[colnames])
    scaled_data['X_validation'] = X_validation_scaled   
    
    X_test_scaled = X_test.copy()
    X_test_scaled[colnames] = sc.fit_transform(X_test[colnames])
    scaled_data['X_test'] = X_test_scaled
    return scaled_data