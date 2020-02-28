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
      0	apple
      1	banana
      2	pear
  """



def one_hot(encodable_df):
    """
    One-hot encodes features of categorical type

    Parameters
    ----------
    encodable_df : pandas.core.frame.DataFrame
        A dataframe of categorical features
    
    Returns
    -------
    pandas.core.frame.DataFrame
        Returns the same dataframe with useful column names and one-hot encoded features

    Examples
    --------
    >>> from prepPy import prepPy as pp
    >>> my_data = pd.DataFrame(np.array([['monkey'], ['dog'], ['cat']]),
                                columns=['animals'])
    >>> pp.one_hot(my_data)
    animals_monkey    animals_dog   animals_cat
            1               0           0
            0               1           0
            0               0           1


    """