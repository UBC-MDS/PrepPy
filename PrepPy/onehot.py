from sklearn.preprocessing import OneHotEncoder

def one_hot(cols, train, valid, test):
    """
    One-hot encodes features of categorical type
    
    Arguments:
    ---------
    cols : list
        list of column names
        
    train : pandas.DataFrame
        The train set from which the columns come
        
    valid : pandas.DataFrame
        The validation set from which the columns come
        
    test : pandas.DataFrame
        The test set from which the columns come
        
    Returns
        train_encoded, valid_encoded, test_encoded : pandas.DataFrames
            The encoded DataFrames

    Examples
    --------
    >>> from prepPy import prepPy as pp
    >>> my_data = pd.DataFrame(np.array([['monkey'], ['dog'], ['cat']]),
                                columns=['animals'])
    >>> pp.one_hot(my_data)
    animals_monkey    animals_dog   animals_cat
            1               0           0
            0               1           0
            0               0           1
    """
    ohe = OneHotEncoder(sparse=False)
    
    names = []
    
    for i in train[cols].columns:
        for j in np.sort(train[i].unique()):
            names.append(i + str(j))
            
    train_encoded = pd.DataFrame(ohe.fit_transform(train[cols]), columns = names)
    valid_encoded = pd.DataFrame(ohe.transform(valid[cols]), columns = names)
    test_encoded = pd.DataFrame(ohe.transform(test[cols]), columns = names)
    
    return train_encoded, valid_encoded, test_encoded