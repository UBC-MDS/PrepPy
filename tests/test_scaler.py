from PrepPy import scaler

import numpy as np

import pandas as pd

import pytest

from pytest import raises


@pytest.fixture
def input_data():
    dataset = {}
    x_train = pd.DataFrame(np.array([['Blue', 56, 4], ['Red', 35, 6],
                                     ['Green', 18, 9]]),
                           columns=['color', 'count', 'usage'])
    dataset['x_train'] = x_train
    x_test = pd.DataFrame(np.array([['Blue', 66, 6], ['Red', 42, 8],
                                    ['Green', 96, 0]]),
                          columns=['color', 'count', 'usage'])
    dataset['x_test'] = x_test
    x_validation = pd.DataFrame(np.array([['Blue', 30, 18], ['Red', 47, 2],
                                          ['Green', 100, 4]]),
                                columns=['color', 'count', 'usage'])
    dataset['x_validation'] = x_validation
    colnames = ['count', 'usage']
    dataset['colnames'] = colnames
    df_empty = pd.DataFrame()  # empty dataframe
    dataset['df_empty'] = df_empty
    wrong_type = np.zeros(6)
    dataset['wrong_type'] = wrong_type
    return dataset


def test_output(input_data):
    # Test data
    x_train = input_data['x_train']
    x_test = input_data['x_test']
    x_validation = input_data['x_validation']
    colnames = input_data['colnames']
    df_empty = input_data['df_empty']
    wrong_type = input_data['wrong_type']
    assert np.equal(round(scaler.scaler(x_train,
                                        x_validation,
                                        x_test,
                                        colnames)
                    ['x_train']['usage'][0], 5),
                    round(-1.135549947915338, 5))
    assert np.equal(round(scaler.scaler(x_train,
                                        x_validation,
                                        x_test,
                                        colnames)
                    ['x_test']['usage'][2], 4),
                    round(-1.3728129459672882, 4))

    assert np.equal(scaler.scaler(x_train, x_validation, x_test, colnames)[
        'x_train'].shape, x_train.shape).all()
    assert np.equal(scaler.scaler(x_train, x_validation, x_test, colnames)[
        'x_test'].shape, x_test.shape).all()
    assert np.equal(scaler.scaler(x_train, x_validation, x_test, colnames)[
        'x_validation'].shape, x_validation.shape).all()
    assert x_train.equals(scaler.scaler(
        x_train, x_validation, x_test, colnames)['x_train']) is False
    assert x_validation.equals(scaler.scaler(
        x_train, x_validation, x_test, colnames)['x_validation']) is False
    assert x_test.equals(scaler.scaler(
        x_train, x_validation, x_test, colnames)['x_test']) is False
    # assert Exception
    with raises(ValueError, match="Input data cannot be empty"):
        scaler.scaler(df_empty, x_validation, x_test, colnames)
    with raises(TypeError, match="A wrong data type has been passed. Please " +
                "pass a dataframe"):
        scaler.scaler(wrong_type, x_validation, x_test, colnames)
    with raises(TypeError, match="Numeric column names is not in a list " +
                "format"):
        scaler.scaler(x_train, x_validation, x_test, x_train)
