import pytest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import scaler

@pytest.fixture

def input_data():
    dataset = {}
    X_train = pd.DataFrame(np.array([['Blue', 56, 4], ['Red', 35, 6], ['Green', 18, 9]]),
                                    columns=['color', 'count', 'usage'])
    dataset['X_train'] =   X_train  

    X_test = pd.DataFrame(np.array([['Blue', 66, 6], ['Red', 42, 8], ['Green', 96, 0]]),
                                    columns=['color', 'count', 'usage'])
    dataset['X_test'] =   X_test
    
    X_validation = pd.DataFrame(np.array([['Blue', 30, 18], ['Red', 47, 2], ['Green', 100, 4]]),
                                     columns=['color', 'count', 'usage'])
    
    dataset['X_validation'] =   X_validation

    colnames = ['count', 'usage']  
    dataset['colnames'] =   colnames
    
    return dataset
 
def test_output(input_data):
    #Test data
    X_train = input_data['X_train']
    X_test = input_data['X_test']
    X_validation = input_data['X_validation']
    colnames = input_data['colnames']


    assert scaler(X_train, X_validation, X_test, colnames)['X_train']['usage'][0] == -1.135549947915338

# def x_train_shape(X_train):
#     assert scaler(X_train, X_validation, X_test, colnames)['X_train'].shape == X_train.shape

# def x_test_shape(X_train):
#     assert scaler(X_train, X_validation, X_test, colnames)['X_test'].shape == X_test.shape

# def x_validation_shape(X_train):
#     assert scaler(X_train, X_validation, X_test, colnames)['X_validation'].shape == X_validation.shape

# def not_equal_output(X_train, X_validation, X_test):
#     assert X_train.equals(scaler(X_train, X_validation, X_test, colnames)['X_train']) == False
#     assert X_validation.equals(scaler(X_train, X_validation, X_test, colnames)['X_validation']) == False
#     assert X_test.equals(scaler(X_train, X_validation, X_test, colnames)['X_test']) == False