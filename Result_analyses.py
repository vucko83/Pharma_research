from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


df=pd.read_csv('new_results_log.csv')
df.shape


def prepare_data(data):
    Y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    n_samples = X.shape[0]
    m_features = X.shape[1]
    return (X, Y, n_samples, m_features)

'''
Starting from log file provided by 
'''


'''
extract the name of the algorithm
'''
df.param_classify.str.split(pat='(')[0]
df['Algorithm']=df.param_classify.map(lambda x: x.split('(')[0])

'''
Group by RMSE and select Algorithms with max RMSE
'''
rmse_set=df.groupby(by=df.Algorithm)['mean_test_neg_mean_squared_error'].agg('max') # group by algorithm name and aggregate on max
idx = df.groupby(['Algorithm'])['mean_test_neg_mean_squared_error'].transform(max) == df['mean_test_neg_mean_squared_error'] #Indices for best RMSE
best_rmse=df[idx]
best_rmse.columns # check columns
best_rmse.shape # check shape
best_rmse.to_csv('new_results_log.csv') # store log with best algorithms by RMSE

'''
Filter performances about splits (within cross validation)
'''
without_splits=best_rmse.filter(regex='^(?!split)', axis=1) # create dataframe and filter results from splits
without_splits=without_splits.filter(regex='^(?!Unnamed)', axis=1)
#without_splits=without_splits.drop([12,24,36,48], axis=0) # If needed drop duplicate algorithms (that have the same RMSE)
#without_splits.to_csv('results_best_rmse_without_10.csv') # if needed store reduced set to csv


'''
Prepare dataframe mean_test_neg_mean_squared_error to RMSE etc
'''
df=df.filter(regex='^(?!Unnamed)', axis=1)
df=df.filter(regex='^(?!split)', axis=1)
df.columns
df['mean_test_RMSE']=np.sqrt(-df.mean_test_neg_mean_squared_error)
df=df.drop([1,2,3,4,10,11,12], axis=0)

'''
Read parameters and recreate models
'''

lasso={'name':'Lasso', 'algorithm':Lasso(alpha=0.01, max_iter=100000), 'features':PCA(n_components=10)}
svr={'name':'SVR', 'algorithm':SVR(C=10, kernel='rbf'),'features':PCA(n_components=10)}
rf={'name':'RF', 'algorithm':RandomForestRegressor(max_depth=10, max_features=22, min_samples_leaf=0.01, n_estimators=60), 'features':PCA(n_components=30)}
ann={'name':'ANN','algorithm':MLPRegressor(learning_rate='constant', momentum=0.4, max_iter=50000, hidden_layer_sizes=(16,)), 'features':PCA(n_components=2)}
lr={'name':'lr','algorithm':LinearRegression(), 'features':NMF(n_components=10)}
ridge={'name':'ridge','algorithm':Ridge(alpha=0.01, max_iter=100000), 'features':PCA(n_components=15)}
gbt={'name':'GBT', 'algorithm':GradientBoostingRegressor(max_depth=2, max_features=7, min_samples_leaf=0.01, n_estimators=90), 'features':SelectKBest(k=30, score_func=mutual_info_regression)}
knn={'name':'K-nn','algorithm':KNeighborsRegressor(n_neighbors=4), 'features':SelectKBest(k=30)}



from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error



setup =[lasso, svr, rf, ann, lr, ridge, gbt, knn]
scoring = ['neg_mean_squared_error', 'r2', 'explained_variance', 'neg_mean_absolute_error','neg_median_absolute_error']

data=pd.read_csv('Aripiprazol_2.csv')
#data=data[data.k<10]
X,y, n_samples, m_features=prepare_data(data)



df_results=pd.DataFrame()
'''
Predictions
'''
predictions={}
for set in setup:
    name = set['name']
    algorithm= set['algorithm']
    features = set['features']
    pipe = Pipeline([
        ('normalize', MinMaxScaler()),
        ('reduce_dim', features),
        ('classify', algorithm)
    ])
    y_hat = cross_val_predict(pipe, X, y, cv=10)
    predictions.update({name:y_hat})


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error


'''
Validation
'''
res={}
for name, pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2=r2_score(y, pred)
    evs=explained_variance_score(y, pred)
    mae=mean_absolute_error(y, pred)
    med_ae=median_absolute_error(y, pred)
    res.update({name:{'RMSE':rmse, 'r2':r2, 'EV':evs, 'MAE':mae, 'MED_AE':med_ae}})

results=pd.DataFrame(res).transpose()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('')
plt.show()

'''
Multiplots
'''

plt.figure(1)
plt.subplot(211)
plt.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
plt.title('GBT')

plt.subplot(212)
plt.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
plt.title('ANN')


plt.subplot(213)
plt.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
plt.title('ANN')

plt.subplot(214)
plt.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
plt.title('ANN')


plt.show()
import seaborn as sns

#https://matplotlib.org/examples/pylab_examples/subplots_demo.html

fig, axarr = plt.subplots(ncols=3, nrows=3)
sns.regplot(x=y, y=predictions['SVR'], ax=axarr[0,0])
sns.regplot(x=y, y=predictions['GBT'], ax=axarr[0,1])
sns.regplot(x=y, y=predictions['RF'], ax=axarr[0,2])
sns.regplot(x=y, y=predictions['ANN'], ax=axarr[1,0])
sns.regplot(x=y, y=predictions['lr'], ax=axarr[1,1])
sns.regplot(x=y, y=predictions['ridge'], ax=axarr[1,2])
sns.regplot(x=y, y=predictions['Lasso'], ax=axarr[2,0])
sns.regplot(x=y, y=predictions['K-nn'], ax=axarr[2,1])



lasso={'name':'Lasso', 'algorithm':Lasso(alpha=0.01, max_iter=100000), 'features':PCA(n_components=10)}
svr={'name':'SVR', 'algorithm':SVR(C=10, kernel='rbf'),'features':PCA(n_components=10)}
rf={'name':'RF', 'algorithm':RandomForestRegressor(max_depth=10, max_features=22, min_samples_leaf=0.01, n_estimators=60), 'features':PCA(n_components=30)}
ann={'name':'ANN','algorithm':MLPRegressor(learning_rate='constant', momentum=0.4, max_iter=50000, hidden_layer_sizes=(16,)), 'features':PCA(n_components=2)}
lr={'name':'lr','algorithm':LinearRegression(), 'features':NMF(n_components=10)}
ridge={'name':'ridge','algorithm':Ridge(alpha=0.01, max_iter=100000), 'features':PCA(n_components=15)}
gbt={'name':'GBT', 'algorithm':GradientBoostingRegressor(max_depth=2, max_features=7, min_samples_leaf=0.01, n_estimators=90), 'features':SelectKBest(k=30, score_func=mutual_info_regression)}
knn={'name':'K-nn','algorithm':KNeighborsRegressor(n_neighbors=4), 'features':SelectKBest(k=30)}




ax1.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))



ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
ax1.set_xlabel('Measured')
ax1.set_ylabel('Predicted')
plt.title('')

ax2.scatter(y, predictions['GBT'], edgecolors=(0, 0, 0))
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
ax2.set_xlabel('Measured')
ax2.set_ylabel('Predicted')
plt.title('')

df_predictions=pd.DataFrame(predictions)
df_predictions.columns
df_predictions['y']=y
df_predictions

df_predictions.columns


predictions
g=sns.PairGrid(data,
               x_vars=['ANN', 'GBT', 'K-nn', 'Lasso', 'RF', 'SVR', 'lr', 'ridge'],
               y_vars=['y']
               )
g=g.map(plt.scatter)

sns.set()
b = sns.regplot(x="y", y="GBT", data=df_predictions,
                 x_estimator=np.mean)
ax


#https://seaborn.pydata.org/generated/seaborn.regplot.html
#https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid
g = sns.JointGrid(x="y", y="SVR", data=df_predictions, space=0)
g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
g = g.plot_marginals(sns.kdeplot, shade=True)

g.

'''
pipe = Pipeline([
        ('normalize', MinMaxScaler()),
        ('reduce_dim', gbt['features']),
        ('classify', gbt['algorithm'])
    ])


r2_score(y,y_hat)

'''










