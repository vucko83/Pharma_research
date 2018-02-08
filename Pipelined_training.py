from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression



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



def prepare_data(data):
    Y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    n_samples = X.shape[0]
    m_features = X.shape[1]
    return (X, Y, n_samples, m_features)


pipe = Pipeline([
    ('normalize', MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('classify', SVR())
])

data=pd.read_csv('Aripiprazol_2.csv')
#data=data[data.k<10]
X,y, n_samples, m_features=prepare_data(data)



scoring = ['neg_mean_squared_error', 'r2', 'explained_variance', 'neg_mean_absolute_error','neg_median_absolute_error']
grid = GridSearchCV(pipe, cv=10, scoring=scoring, refit=scoring[0], n_jobs=4, param_grid=params_dicts_all, return_train_score=True)
grid.fit(X, np.log(y))
df=pd.DataFrame(grid.cv_results_)
df.to_csv('test_results_new_log.csv')


'''
X.describe()
grid.best_estimator_
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
np.average(cross_val_score(grid.best_estimator_,X,y=y,scoring='r2', n_jobs=4, cv=5))


'''
