import numpy as np
from sklearn.model_selection import train_test_split

def train_valid_test_split(X, y, test_size=None, valid_size=None, train_size=None, stratify=None, random_state=None, shuffle=True):
    """
    Split arrays or matrices into random train, validation and test subsets
    
    Parameters
    ----------
    X, y: Sequence of indexables of the same length / shape[0]
    Allowable inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes
    
    test_size: float or None, optional (default=None)
        If float, a value between 0.0 and 1.0 to represent the proportion of the input dataset to
          comprise the size of the test subset
        If None, the value is set to the complement of the sum of the train_size and valid_size
        If train_size and valid_size are also None, it will be set to 0.25
      
    valid_size: float or None, (default=None)
        If float, a value between 0.0 and 1.0 to represent the proportion of the input dataset to
          comprise the size of the validation subset
        If None, the value will be set to 0.0
      
    train_size: float or None, (default=None)
        If float, a value between 0.0 and 1.0 to represent the proportion of the input dataset to
          comprise the size of the train subset
        If None, the value will be set to the complement of the test_size and valid_size
  
    stratify: array-like or None (default=None)
        If not None, splits categorical data in a stratified fashion preserving the same proportion of
            classes in the train, valid and test sets, using this input as the class labels    
      
    random_state: integer, optional (default=None)
        A value for the seed to be used by the random number generator
        If None, the value will be set to 1
      
    shuffle: logical, optional (default=TRUE)
        Indicate whether data is to be shuffled prior to splitting
  
    Returns
    -------
    splits: list, length = 3 * len(arrays)
        List containing train, validation and test splits of the input data
      
    Examples
    --------
    >>> from PrepPy import PrepPy as pp
    >>> X, y = np.arange(16).reshape((8, 2)), list(range(8))
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
    
    >>> X_train, X_valid, X_test, y_train, y_valid, y_test = pp.train_valid_test_split(X, y, test_size=0.25, valid_size=0.25, random_state=777)

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
  
  
    >>> pp.train_valid_test_split(X, test_size=2, shuffle=False)

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

    # Check data input types and parameters
    assert ((isinstance(X, pd.DataFrame)) or (isinstance(X, np.ndarray)) or (isinstance(X, pd.DataFrame))), "Please input a list, numpy array, scipy-sparse matrices or pandas dataframe"
    assert ((isinstance(y, pd.DataFrame)) or (isinstance(X, np.ndarray)) or (isinstance(X, pd.DataFrame))), "Please input a list, numpy array, scipy-sparse matrices or pandas dataframe"

    assert len(X) != 0, "Your input is empty"
    assert len(y) != 0, "Your input is empty"
    
    # Initialize splitting ratios for `train` and `valid` sets
    train_ratio = 0
    valid_ratio = 0
    
    # Set `valid_size` = 0 if `valid_size` = None
    if valid_size == None:
        valid_size = 0.0
    
    # Split into `test` set and `resplit` set to be resplit into `train` and `valid` sets 
    X_resplit, X_test, y_resplit, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=None, 
                                                            train_size=None,
                                                            stratify=None,
                                                            random_state=1,
                                                            shuffle=True)
    
    # Calculate ratios for second call to `train_test_split` to produce `valid` and `train` sets 
    valid_ratio = valid_size / (1 - test_size)
    train_ratio = 1 - valid_ratio
        
    X_train, X_valid, y_train, y_valid  = train_test_split(X_resplit,
                                                           y_resplit, 
                                                           test_size=valid_ratio, 
                                                           train_size=train_ratio,
                                                           stratify=None,
                                                           random_state=1,
                                                           shuffle=True)
                                                           
    return X_train, X_valid, y_train, y_valid, X_test, y_test
