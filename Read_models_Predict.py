import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
np.set_printoptions(suppress=True)

y_hat=cross_val_predict(grid, pd.DataFrame(X), np.log(y), cv=10)

y_hat_exp=np.exp(y_hat)
r2_score(y,y_hat_exp)

y_hat_exp




grid.cv_results_['params'][0]

grid.cv_results_['mean_train_r2']



par=grid.cv_results_['params'][0]

par_grid={key:[value] for key, value in par.items()}

a.best_estimator_.predict(X,y)



from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

par
pipe1 = Pipeline([
    ('normalize',  MinMaxScaler()),
    ('reduce_dim', SelectKBest()),
    ('classify', Lasso())
])


pipe.get_params()

pipe.set_params()

a=GridSearchCV(pipe, cv=2, scoring=scoring, refit=scoring[0], n_jobs=2, param_grid=par_grid, return_train_score=True).fit(X,y)


cross_val_score(a, X,y=y, scoring='r2', n_jobs=2,  cv=5)

cross_val_predict(a, X,y=y, n_jobs=2,  cv=5)


