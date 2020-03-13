from PrepPy import onehot
import pytest
import pandas as pd
import numpy as np

helperdata1 = pd.DataFrame(np.array([['monkey'], ['dog'], ['cat']]),
                                columns=['animals'])


def one_hot_test1():
    output = one_hot(cols = ['animals'], train = helperdata1)
    assert(len(output.columns) == helperdata1.shape[0])
    
def one_hot_test2():
    output = one_hot(cols = ['animals'], train = helperdata1, valid = helperdata1)
    assert(len(output) == 2)
    
def one_hot_test3():
    output = one_hot(cols = ['animals'], train = helperdata1, valid = helperdata1, test = helperdata1)
    assert(len(output) == 3)