from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def onehot(cols, train, valid=None, test=None):
    """
    One-hot encodes features of categorical type

    Arguments:
    ---------
    cols : list
        list of column names

    train : pandas.DataFrame
        The train set from which the columns come

    valid : pandas.DataFrame
        The validation set from which the columns come

    test : pandas.DataFrame
        The test set from which the columns come

    Returns
        train_encoded, valid_encoded, test_encoded : pandas.DataFrames
            The encoded DataFrames

    Examples
    --------
    >>> from prepPy import prepPy as pp
    >>> my_data = pd.DataFrame(np.array([['monkey'], ['dog'], ['cat']]),
                                columns=['animals'])
    >>> pp.onehot(pp.onehot(['animals'], mydata))
    animals_monkey    animals_dog   animals_cat
            1               0           0
            0               1           0
            0               0           1
    """
    ohe = OneHotEncoder(sparse=False)

    names = []

    for i in train[cols].columns:
        for j in np.sort(train[i].unique()):
            names.append(i + '_' + str(j))

    train_encoded, valid_encoded, test_encoded = (None, None, None)

    train_encoded = pd.DataFrame(ohe.fit_transform(train[cols]),
                                 columns=names)

    if valid is not None:
        valid_encoded = pd.DataFrame(ohe.transform(valid[cols]),
                                     columns=names)

        # Try-except for data_type
        if not isinstance(valid, pd.DataFrame):
            raise Exception("Please provide a valid Pandas DataFrame object")
        elif len(valid) == 0:
            raise Exception("Your 'valid' DataFrame is empty")

    if test is not None:
        test_encoded = pd.DataFrame(ohe.transform(test[cols]),
                                    columns=names)

        # Try-except for data_type
        if not isinstance(test, pd.DataFrame):
            raise Exception("Please provide a valid Pandas DataFrame object")
        elif len(test) == 0:
            raise Exception("Your 'test' DataFrame is empty")

    return train_encoded, valid_encoded, test_encoded
