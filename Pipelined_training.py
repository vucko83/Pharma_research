from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
import pandas as pd
import numpy as np

#http://scikit-learn.org/stable/modules/pipeline.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

data=pd.read_csv('Aripiprazol.csv')

def prepare_data(data):
    Y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    #scaler = StandardScaler()
    #scaler.fit(X)
    #X = scaler.transform(X)
    n_samples = X.shape[0]
    m_features = X.shape[1]
    return (X, Y, n_samples, m_features)

'''
pca=PCA(n_components=2)
model=pca.fit_transform(X)

model.
X,y, n_samples, m_features=prepare_data(data)
X.describe

'''
from sklearn.ensemble import IsolationForest
X,y, n_samples, m_features=prepare_data(data)

pipe = Pipeline([
    ('normalize', MinMaxScaler()),
    #('outliers', IsolationForest),
    ('reduce_dim', PCA()),
    ('classify', SVR())
])

from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


'''
N_FEATURES_OPTIONS = [2, 4, 8, 16, 28]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        # 'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim': [PCA(), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },

    {
        'reduce_dim': [SelectKBest(mutual_info_regression)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    }

]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

'''




scoring = ['neg_mean_squared_error', 'r2']
grid = GridSearchCV(pipe, cv=10, scoring=scoring, refit=scoring[0], n_jobs=2, param_grid=a, return_train_score=True)
grid.fit(X, y).predict(X)
grid.cv_results_


X.describe()
grid.best_estimator_
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
np.average(cross_val_score(grid.best_estimator_,X,y=y,scoring='r2', n_jobs=4, cv=5))


'''


  '''
