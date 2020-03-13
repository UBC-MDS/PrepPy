import pytest
from pytest import raises
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
from PrepPy import scaler
import unittest

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
    
    df_empty = pd.DataFrame() #empty dataframe
    dataset['df_empty'] = df_empty

    wrong_type = np.zeros(6)
    dataset['wrong_type'] = wrong_type

    return dataset
 
def test_output(input_data):
    #Test data
    X_train = input_data['X_train']
    X_test = input_data['X_test']
    X_validation = input_data['X_validation']
    colnames = input_data['colnames']
    df_empty = input_data['df_empty'] 
    wrong_type = input_data['wrong_type']

    assert np.isclose(scaler.scaler(X_train, X_validation, X_test, colnames)['X_train']['usage'][0], -1.135549947915338) == True
    assert np.isclose(scaler.scaler(X_train, X_validation, X_test, colnames)['X_test']['usage'][2], -1.3728129459672882) == True

    assert scaler.scaler(X_train, X_validation, X_test, colnames)['X_train'].shape == X_train.shape
    assert scaler.scaler(X_train, X_validation, X_test, colnames)['X_test'].shape == X_test.shape
    assert scaler.scaler(X_train, X_validation, X_test, colnames)['X_validation'].shape == X_validation.shape

    assert X_train.equals(scaler.scaler(X_train, X_validation, X_test, colnames)['X_train']) == False
    assert X_validation.equals(scaler.scaler(X_train, X_validation, X_test, colnames)['X_validation']) == False
    assert X_test.equals(scaler.scaler(X_train, X_validation, X_test, colnames)['X_test']) == False

    # #assert Exception
    # with raises(ValueError, match = "Input data cannot be empty"):
    #   scaler.scaler(df_empty, X_validation, X_test, colnames)

    # with raises(TypeError, match = "A wrong data type has been passed. Please pass a dataframe"):
    #   scaler.scaler(wrong_type, X_validation, X_test, colnames)

    # with raises(TypeError, match = "Numeric column names is not in a list format"):
    #   scaler.scaler(X_train, X_validation, X_test, X_train)