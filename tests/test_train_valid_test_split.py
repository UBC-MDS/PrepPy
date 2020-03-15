from PrepPy import train_valid_test_split

import numpy as np

import pytest


# Check data input types and parameters

X, y = np.arange(16).reshape((8, 2)), list(range(8))


def test_train_test_valid_split():
    """
    This script will test the output of the train_valid_test_split function
    which splits dataframes into random train, validation and test subsets.
    The proportion of the train set relative to the input data will be
    equal to valid_size * (1 - test_size).
    """

    X_train, X_valid, X_test, y_train, y_valid, y_test =\
        train_valid_test_split.train_valid_test_split(X, y)

    assert(len(X_train) == 4)
    assert(len(X_valid) == 2)
    assert(len(X_test) == 2)


def check_exception():

    with pytest.raises(Exception):
        train_valid_test_split.train_valid_test_split("test", y)

    with pytest.raises(Exception):
        train_valid_test_split.train_valid_test_split(X, "test")

test_train_test_valid_split()
check_exception()
