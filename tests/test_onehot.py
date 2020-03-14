from PrepPy import onehot
import pandas as pd
import numpy as np

helperdata1 = pd.DataFrame(np.array([['monkey'],
                                     ['dog'],
                                     ['cat']]),
                           columns=['animals'])


def onehot_test1():
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1)
    assert(output['train'].shape == (3, 3))


def onehot_test2():
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1)
    assert(output['valid'].shape == (3, 3))


def onehot_test3():
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1,
                           test=helperdata1)
    assert(len(output) == 3)
    
def check_exception1():
    with pytest.raises(Exception):
        onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid="helperdata1")
                           
def check_exception2():
    with pytest.raises(Exception):
        onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1,
                           test="helper")
