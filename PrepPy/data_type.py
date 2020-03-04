def data_type(df):
    """
    Identify features of different data types.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Original feature dataframe containing one column for each feature.

    Returns
    -------
    dict
        Stores the categorical and numerical columns separately as two dataframes.

    Examples
    --------
    >>> from prepPy import prepPy as pp
    >>> my_data = pd.DataFrame(np.array([['apple', 3, 0], ['banana', 5, 6], ['pear', 8, 9]]),
                               columns=['fruits', 'count', 'price'])
    >>> pp.data_type(my_data)['num']
          count price
        0     3     0
        1     5     6
        2     8     9
    >>> pp.data_type(my_data)['cat']
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
