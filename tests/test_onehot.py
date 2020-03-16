from preppy524 import onehot
import pandas as pd
import numpy as np

helperdata1 = pd.DataFrame(np.array([['monkey'],
                                     ['dog'],
                                     ['cat']]),
                           columns=['animals'])


def onehot_test1():
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1)
    assert(len(output.columns) == helperdata1.shape[0])


def onehot_test2():
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1)
    assert(len(output) == 2)


def onehot_test3():
    output = onehot.onehot(cols=['animals'],
                           train=helperdata1,
                           valid=helperdata1,
                           test=helperdata1)
    assert(len(output) == 3)
