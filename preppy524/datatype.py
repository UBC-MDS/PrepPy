import pandas as pd


def data_type(df):
    """
    Identify features of different data types.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Original feature dataframe containing one column for each feature.

    Returns
    -------
    tuple
        Stores the categorical and numerical columns separately
        as two dataframes.

    Examples
    --------
    >>> from PrepPy import datatype
    >>> my_data = pd.DataFrame({'fruits': ['apple', 'banana', 'pear'],
                                'count': [3, 5, 8],
                                'price': [1.0, 6.5, 9.23]})
    >>> datatype.data_type(my_data)[0]
          count price
        0     3     1.0
        1     5     6.5
        2     8     9.23
    >>> datatype.data_type(my_data)[1]
          fruits
        0   apple
        1   banana
        2   pear
    """

    # Try-except for data_type
    if not isinstance(df, pd.DataFrame):
        raise Exception("Please provide a valid Pandas DataFrame object")
    elif len(df) == 0:
        raise Exception("Your DataFrame is empty")

    cols = df.columns
    numeric_vars = df._get_numeric_data().columns
    categorical_vars = [c for c in cols if c not in numeric_vars]

    return (df[numeric_vars], df[categorical_vars])
