from preppy524 import onehot
import pandas as pd
import numpy as np

import pandas as pd

import pytest

helperdata1 = pd.DataFrame(np.array([['monkey'],
                                     ['dog'],
                                     ['cat']]),
                           columns=['animals'])


def onehot_test1():
    # test if train set has been encoded correctly
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1)
    assert(output['train'].shape == (3, 3))


def onehot_test2():
    # test if validation set has been encoded correctly
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1)
    assert(output['valid'].shape == (3, 3))


def onehot_test3():
    # test if output is correct when three sets are passed to the function
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1,
                           test=helperdata1)
    assert(len(output) == 3)


def check_exception1():
    # check exception handling when validation set input is of wrong data type
    with pytest.raises(Exception):
        onehot.onehot(cols=['animals'],
                      train=helperdata1,
                      valid="helperdata1")


def check_exception2():
    # check exception handling when test set input is of wrong data type
    with pytest.raises(Exception):
        onehot.onehot(cols=['animals'],
                      train=helperdata1,
                      valid=helperdata1,
                      test="helper")


def check_exception3():
    # check exception handling when validation set input is an empty data frame
    with pytest.raises(Exception):
        onehot.onehot(cols=['animals'],
                      train=helperdata1,
                      valid=pd.DataFrame())


def check_exception4():
    # check exception handling when test set input is an empty data frame
    with pytest.raises(Exception):
        onehot.onehot(cols=['animals'],
                      train=helperdata1,
                      valid=helperdata1,
                      test=pd.DataFrame())

onehot_test1()
onehot_test2()
onehot_test3()
check_exception1()
check_exception2()
check_exception3()
check_exception4()
