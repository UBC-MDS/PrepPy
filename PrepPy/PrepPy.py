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

def scaler(X_train, X_test, num_columns):
  """
  Perform standard scaler on numerical features. 
  
  Parameters
  ----------
  X_train : pandas.core.frame.DataFrame
    Dataframe of train set containing columns to be scaled.
  
  X_test : pandas.core.frame.DataFrame
    Dataframe of test set containing columns to be scaled.  

  num_columns : list
    A list of numeric column names
  
  Returns
  -------
  dict
      Stores the scaled and transformed X_train and X_test sets separately as two dataframes.

 Examples
  --------
  >>> from prepPy import prepPy as pp

  >>> x_train = pd.DataFrame(np.array([['Blue', 56, 4], ['Red', 35, 6], ['Green', 18, 9]]),
                             columns=['color', 'count', 'usage'])

  >>> x_test = pd.DataFrame(np.array([['Blue', 66, 6], ['Red', 42, 8], ['Green', 96, 0]]),
                             columns=['color', 'count', 'usage'])

  >>> colnames = ['count', 'usage']                          

  >>> X_train = pp.scaler(x_train, x_test, colnames)['x_train']

  >>> X_train    
    color	count	usage
0	Blue	1.26538	-1.13555
1	Red	    -0.0857887	-0.162221
2	Green	-1.17959	1.29777

  >>> X_test = pp.scaler(x_train, x_test, colnames)['x_test']

  >>> X_test
    color	count	usage
 0  Blue    1.90879917 -0.16222142
 1  Red     0.36460209  0.81110711
 2  Green   3.83904552 -3.082207 
  """
