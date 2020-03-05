from PrepPy import PrepPy as pp

import pandas as pd
import numpy as np

helperdata1 = pd.DataFrame(np.array([['monkey'], ['dog'], ['cat']]),
                                columns=['animals'])


def onehot_test1():
    output = pp.onehot(helperdata1)
    assert(output.columns == helperdata1.iloc[1])