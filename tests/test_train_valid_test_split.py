from PrepPy import PrepPy as pp 

import pandas as pd
import numpy as np

X, y = np.arange(16).reshape((8, 2)), list(range(8))

# Check data input types and parameters

def test_train_test_valid_split():
    """
    This script will test the output of the train_valid_test_split function
    which splits arrays or matrices into random train, validation and test subsets
 
    The proportion of the train set relative to the input data will be valid_size * (1 - test_size)
    """
    
    output = pp.train_valid_test_split(X,y)
    
    assert(output['X_train'].shape[0] == 4)
    assert(output['X_valid'].shape[0] == 2)
    assert(output['X_test'].shape[0] == 2)
    assert(output['y_train'].shape[0] == 4)
    assert(output['y_valid'].shape[0] == 2)
    assert(output['y_test'].shape[0] == 2)
    
    
    
