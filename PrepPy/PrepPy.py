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


def train_valid_test(X, y, test_size, valid_size, train_size, stratify, random_state, shuffle)):
  """
  Split arrays or matrices into random train, validation and test subsets
  
  Parameters
  ----------
  X, y: Sequence of indexables of the same length / shape[0]
      Allowable inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes
       
  test_size: float, int or None, optional (default=None)
      If float, a value between 0.0 and 1.0 to represent the proportion of the dataset to
          comprise the size of the test subset
      If int, a value for the absolute number of test samples
      If None, the value is set to the complement of the sum of the train_size and test_size
      If train_size and test_size are also None, it will be set to 0.25
      
  valid_size: float, int or None, (default=None)
      If float, a value between 0.0 and 1.0 to represent the proportion of the dataset to
          comprise the size of the test subset
      If int, a value for the absolute number of test samples
      If None, the value will be set to 0.0
      
  train_size: float, int or None, (default=None)
      If float, a value between 0.0 and 1.0 to represent the proportion of the dataset to
          comprise the size of the test subset
      If int, a value for the absolute number of test samples
      If None, the value will be set to the complement of the test_size and train_size
  
  stratify: array-like or None (default=None)
      If not None, splits categorical data in a stratified fashion preserving the same proportion
          of classes in the train, valid and test sets, using this input as the class labels    
      
  random_state: int, optional (default=None)
      A value for the seed to be used by the random number generator
      If None, the value will be set to `123`
      
  shuffle: boolean, optional (default=True)
      Indicate whether data is to be shuffled prior to splitting
  
  Returns
  -------
  splits: list, length=3 * len(arrays)
      List containing train, validation and test splits of the input data
      
  Examples
  --------
  >>> from prepPy import prepPy as pp
  >>> X, y = np.arange(16).reshape((8, 2)), range(8)
  >>> X
  array([[0, 1],
         [2, 3],
         [4, 5],
         [6, 7],
         [8, 9],
         [10, 11],
         [12, 13],
         [14, 15]])
           
  >>> list(y)
  [0, 1, 2, 3, 4, 5, 6, 7]
    
  >>> X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split(
            X, y, test_size=0.25, valid_size=0.25, random_state=777)

  >>> X_train
  array([[4, 5],
         [0, 1],
         [6, 7],
         [12, 13]])
           
  >>> y_train
  [3, 0, 2, 5]

  >>> X_valid
  array([[2, 3],
         [10, 11]])
           
  >>> y_valid
  [1, 4]
  
  >>> X_test
  array([[8, 9],
         [14, 15]])
           
  >>> y_test
  [7, 6]  
  
  
  >>> train_test_split(X, test_size=2, shuffle=False)

  >>> X_train
  array([[2, 3],
         [14, 15],
         [6, 7],
         [12, 13],
         [4, 5],
         [10, 11]])
  
  >>> X_test
  array([[8, 9],
         [0, 1]])
         
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
  
  X_valid : pandas.core.frame.DataFrame
    Dataframe of validation set containing columns to be scaled.

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
  >>> from PrepPy import prepPy as pp
  >>> x_train = pd.DataFrame(np.array([['Blue', 56, 4], ['Red', 35, 6], ['Green', 18, 9]]),
                             columns=['color', 'count', 'usage'])

  >>> x_test = pd.DataFrame(np.array([['Blue', 66, 6], ['Red', 42, 8], ['Green', 96, 0]]),
                             columns=['color', 'count', 'usage'])

  >>> x_valid = pd.DataFrame(np.array([['Blue', 30, 18], ['Red', 47, 2], ['Green', 100, 4]]),
                             columns=['color', 'count', 'usage'])

  >>> colnames = ['count', 'usage']                          

  >>> X_train = pp.scaler(x_train, x_test, colnames)['x_train']

  >>> X_train    
    color	count	usage
0	Blue	1.26538	-1.13555
1	Red	  -0.0857887	-0.162221
2	Green	-1.17959	1.29777


  >>> X_valid = pp.scaler(x_train, x_valid, x_test, colnames)['x_valid']

  >>> X_valid
    color	count	usage
 0  Blue    1.80879917 -0.16222142
 1  Red     0.16460209  1.81110711
 2  Green   2.43904552 -4.082207 

  >>> X_test = pp.scaler(x_train, x_test, colnames)['x_test']

  >>> X_test
    color	count	usage
 0  Blue    1.90879917 -0.16222142
 1  Red     0.36460209  0.81110711
 2  Green   3.83904552 -3.082207 
  """
