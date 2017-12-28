import pandas as pd
import pickle as pkl
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
#import xgboost as xgb

from sklearn.svm import SVR

from sklearn import pipeline
from sklearn import metrics

'''
Store preocedure
'''
def store_model(model, name):
    path='Models/'+name+'.pkl'
    file=open(path, 'wb')
    pkl._dump(model,file)


'''
Import data
'''
data=pd.read_csv('Aripiprazol.csv')

#data.shape
#data.head()

'''
Create feature sets
'''

Y=data.iloc[:,-1] # label
X=data.iloc[:,:-1] # input data
#Y=Y[0:30]
#X=X.iloc[0:30,0:5]

X.shape
# 6, 15 features

'''
Define scoring (evaluation measures)
'''
scoring = ['neg_mean_squared_error', 'explained_variance', 'neg_mean_absolute_error', 'r2', 'neg_median_absolute_error']

'''
Define Algorithms
'''
####!!!!!!!! Change number of features or create relative RF and GBT

random_forest_parameters={'max_depth': range(2, 5, 1), 'n_estimators': range(10, 101, 10), 'max_features':range(5,30,5), 'min_samples_leaf': range(2,11,2)}
knn_parameters={'n_neighbors': range(1,11,1)}
gbt_parameters={'max_depth': range(2, 6, 1), 'n_estimators': range(10, 101,10), 'max_features':range(5,30,5), 'min_samples_leaf': range(2,11,2)}
ann_parameters={'learning_rate':['constant'], 'learning_rate_init':[0.1,0.2,0.4,0.6,0.8,0.9], 'momentum':[0.1,0.2,0.4,0.6,0.8,0.9], 'max_iter':[50000]}
lasso_parameters={'alpha':[0.1,0.2,0.4,0.6,0.8,0.9], 'max_iter':[10000]}
lr_parameters={'normalize':[True]}

models=[]
models.append(('K-nn', KNeighborsRegressor(), knn_parameters))
models.append(('Random_Forest', RandomForestRegressor(random_state=42), random_forest_parameters))
models.append(('GBT', GradientBoostingRegressor(random_state=42), gbt_parameters))
models.append(('ANN', MLPRegressor(), ann_parameters))
models.append(('Lasso',Lasso(), lasso_parameters))
models.append(('LR',LinearRegression(), lr_parameters))

'''
Optimize Models
'''
i=0
results=[]
for name, model, params in models:
    gs = GridSearchCV(model,
                  param_grid=params,
                  scoring=scoring, cv=10,  return_train_score=True, refit=scoring[0], n_jobs=2)
    gs.fit(X, Y)
    print(i+1,name)
    result=pd.DataFrame(gs.cv_results_).filter(regex='^(?!split)', axis=1) # create dataframe and filter results from splits
    result = result.filter(regex='^(?!param_)', axis=1)
    vec=result['rank_test_neg_mean_squared_error']==1
    result=result[vec]
    result['Name']=name  # add name of algorithm to log
    results.append((name, result , gs.best_estimator_, {name:gs.best_params_}))


df_predictions=pd.DataFrame(Y) #Creates initial df with ground truth
full_results=[]
best_params_full={}
for name, result, best_estimator, best_params in results: #adds
    store_model(best_estimator, name)
    full_results.append(result)
    df_predictions[name]=best_estimator.predict(X)
    best_params_full.update(best_params)



import json

with open('best_params.json', 'w') as outfile:
    json.dump(best_params_full, outfile)

df_results=pd.concat(full_results)
df_results.to_csv('results.csv')
df_predictions.to_csv('predictions.csv')


'''

'''









'''
Run experiments
'''

'''
### SVR

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


'''

'''
Pipelining
'''

'''
clf = pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)

'''

