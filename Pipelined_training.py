from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

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

'''
Metrics
'''

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error



def prepare_data(data):
    Y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    n_samples = X.shape[0]
    m_features = X.shape[1]
    return (np.array(X), np.array(Y), n_samples, m_features)


pipe = Pipeline([
    ('normalize', MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('classify', SVR())
])

data=pd.read_csv('Aripiprazol_2.csv')
#data=data[data.k<10]
X,y, n_samples, m_features=prepare_data(data)



'''
Scoring functions
'''

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error



def unlog(y, y_pred, measure):

    y_original = np.exp(y)
    y_pred_original = np.exp(y_pred)

    score = measure(y_original, y_pred_original)
    return (score)

scoring = {
    'R2': make_scorer(unlog, measure=r2_score ,greater_is_better = True),
    'Explained_Variance': make_scorer(unlog, measure=explained_variance_score, greater_is_better = True),
    'MAE': make_scorer(unlog, measure=mean_absolute_error ,greater_is_better = False),
    'MSE': make_scorer(unlog, measure=mean_squared_error ,greater_is_better = False),
    'MSLE': make_scorer(unlog, measure=mean_squared_log_error ,greater_is_better = False),
    'Median_AE': make_scorer(unlog, measure=median_absolute_error ,greater_is_better = False)
}

grid = GridSearchCV(pipe, cv=10, scoring=scoring, refit='MSE', n_jobs=4, param_grid=params_dicts_all, return_train_score=True)
grid.fit(X, np.log(y))

print(max(grid.cv_results_['mean_test_R2']))



'''
a=cross_val_score(grid.best_estimator_, X,y, cv=10, n_jobs=2)

b=cross_val_predict(grid.best_estimator_, X,np.log(y), cv=10, n_jobs=2)

y_hat_exp=np.exp(b)
r2_score(y,y_hat_exp)


grid.best_estimator_.named_steps['reduce_dim'].get_support()

grid.best_estimator_.named_steps['classify']


df=pd.DataFrame(grid.cv_results_)



df.to_csv('Results/test_results_K_Best_Mutual.csv')

grid.cv_results_['params']

grid.cv_results_


from sklearn.linear_model import LinearRegression
import numpy as np


def score_reg(X, y):
    LR = LinearRegression()
    model = LR.fit(X, y)
    return (np.abs(model.coef_))


from sklearn.feature_selection import SelectKBest

a = SelectKBest(score_func=score_reg, k=5).fit(X, y)

reduced = a.transform(X)
reduced.shape

'''

