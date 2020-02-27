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
