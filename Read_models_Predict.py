import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
data=pd.read_csv('Aripiprazol.csv')

def prepare_data(data):
    y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    return (X, y)

X,y =prepare_data(data)

'''
linear dependance
'''


####### Outlier detection

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

outlier_detectors={
    'Isolation_forest': IsolationForest(contamination=0.1),
    'Eliptic_envelope': EllipticEnvelope(contamination=0.1),
    'One_SVM': OneClassSVM(nu=0.1)
}

for name, outlier in outlier_detectors.items():
    outlier.fit(X)
    outlier_indicators=outlier.predict(X)
    d=data[outlier_indicators==-1]
    d.to_csv('outliers_'+name+'.csv')

one=OneClassSVM(nu=0.5)
one.fit(X)
sel=one.predict(X)
sel


# fit the model
rng = np.random.RandomState(42)
clf = IsolationForest(random_state=rng, contamination=0.1)
clf.fit(X)
y_pred_train = clf.predict(X)




clf=EllipticEnvelope(support_fraction=1.,contamination=0.1)
clf.fit(X)
y_pred_train = clf.predict(X)


X1=X[y_pred_train==1]
y1=y[y_pred_train==1]

len(y1)
np.log(y)

pipe = Pipeline([
    ('normalize', MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('classify', LinearRegression())
])

N_FEATURES_OPTIONS = list(range(2,31,2))
param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,

    }
    ]


np.log(X)

scoring = ['neg_mean_squared_error', 'r2']
grid = GridSearchCV(pipe, cv=10, scoring=scoring, refit=scoring[0], n_jobs=4, param_grid=param_grid, return_train_score=True)
grid.fit(X1, y1)

grid.cv_results_['mean_test_r2']
grid.best_params_

############ Vizualization #############

fig, ax = plt.subplots()
ax.scatter(y, y_hat, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('')
plt.show()
