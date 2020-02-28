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