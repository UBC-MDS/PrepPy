from preppy524 import datatype
import pandas as pd
import pytest


test_dict = {'cat1': ['apple', None, 'pear', 'banana', 'blueberry', 'lemon'],
             'num1': [0, 1, 2, 3, 4, 5],
             'cat2': [True, False, False, True, False, None],
             'num2': [0, 16, 7, None, 10, 14],
             'num3': [0.5, 3, 3.9, 5.5, 100.2, 33]}

test_data = pd.DataFrame(test_dict)


def test_datatype_num():
    # test if numeric data is correctly separated from original data
    assert datatype.data_type(test_data)[0].equals(test_data[['num1',
                                                              'num2',
                                                              'num3']])


def test_datatype_cat():
    # test if categorical data is correctly separated from original data
    assert datatype.data_type(test_data)[1].equals(test_data[['cat1',
                                                              'cat2']])


def check_exception_wrong_input():
    # test if an invalid input will be handled by function correctly
    with pytest.raises(Exception):
        datatype.data_type("df")


def check_exception_empty_df():
    # test if an empty input will be handled by function correctly
    with pytest.raises(Exception):
        datatype.data_type(pd.DataFrame())

<<<<<<< HEAD

test_datatype1()
test_datatype2()
check_exception1()
check_exception2()
=======
test_datatype_num()
test_datatype_cat()
check_exception_wrong_input()
check_exception_empty_df()
>>>>>>> 34807fa47b778677a9348a5d3a1f17cd4336c93c
