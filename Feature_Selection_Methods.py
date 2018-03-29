

'''

https://programtalk.com/python-examples/sklearn.feature_selection.GenericUnivariateSelect/
https://www.programcreek.com/python/example/93976/sklearn.feature_selection.SelectFromModel
https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/

''''''

import pandas as pd
from sklearn.feature_selection import GenericUnivariateSelect

from sklearn.feature_selection import SelectFromModel

df=pd.read_csv('Aripiprazol_2.csv')

df.describe()

df.head()


GenericUnivariateSelect


def fit(self, X, y):
    import scipy.sparse
    import sklearn.feature_selection

    self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
        score_func=self.score_func, param=self.alpha, mode=self.mode)

    # Because the pipeline guarantees that each feature is positive,
    # clip all values below zero to zero
    if self.score_func == sklearn.feature_selection.chi2:
        if scipy.sparse.issparse(X):
            X.data[X.data < 0] = 0.0
        else:
            X[X < 0] = 0.0

    self.preprocessor.fit(X, y)
    return self






