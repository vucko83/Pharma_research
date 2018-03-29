import pandas as pd
import pickle as pkl
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#import xgboost as xgb

from sklearn.svm import SVR

from sklearn import pipeline
from sklearn import metrics


'''

Additional metric, check

from revrand.metrics import lins_ccc

lins_ccc(np.array([0.1, 0.3]),np.array([0.3, 0.2]))

revrand.metrics.lins_ccc(y_true, y_pred)

'''

'''
Store preocedure
'''
def store_model(model, name, feature_set):
    path='Models/'+name+'_'+feature_set+'.pkl'
    file=open(path, 'wb')
    pkl._dump(model,file)


'''
Create feature sets
'''
def create_feature_sets(data):
    predictors_6=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'k']
    # Based on corellation
    predictors_15=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'ShpC', 'H-don', 'H-acc', 'logP', 'pol', 'Ecd', 'Ed', 'Torsion E (Et)', 'Total Energy (E)', 'k']
    return ({'6_features':data[predictors_6], '15_features':data[predictors_15], '30_features':data})

def n_features_range(n_samples, m_features, m_n_ratio=1):
    '''
    
    :param n_samples: 
    :param m_features: 
    :param m_n_ratio: 
    :return: 
    '''

    max_features=int(round(n_samples/m_n_ratio))
    if m_features<max_features:
        max_features=m_features

    if m_features==1:
        min_features=1
    else:
        min_features=2

    step_size=int(round(np.log2(max_features)))

    param_range = list(range(min_features, max_features, step_size))

    if param_range[-1]<max_features:
        param_range.append(max_features)

    return param_range

def rf_parameters(n_samples, m_features, m_n_ratio=1):
        random_forest_parameters = {
                                'n_estimators': range(10, 61, 20),
                                'max_features': n_features_range(n_samples,m_features),
                                'min_samples_leaf': np.arange(0.01, 0.06, 0.01),
                                'max_depth': range(2, 11, 2),
                              }

        return (random_forest_parameters)


def nn_size(m_features):
    half=int(np.ceil((m_features+1)/2))
    quarter=int(np.ceil((m_features + 1)/4))
    size=[(quarter,), (half,)]
    return(size)


def create_models(n_samples, m_features):
    random_forest_parameters = rf_parameters(n_samples, m_features)
    gbt_parameters = rf_parameters(n_samples, m_features)
    knn_parameters = {'n_neighbors': range(1, 11, 1)}
    ann_parameters = {'learning_rate': ['constant'], 'learning_rate_init': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
                      'momentum': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], 'max_iter': [10000, 50000],
                      'hidden_layer_sizes': nn_size(m_features)}
    lasso_parameters = {'alpha': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], 'max_iter': [10000, 50000]}
    lr_parameters = {'normalize': [True]}
    svr_parameters = {'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}


    models = []
    '''

    models.append(('RF', RandomForestRegressor(random_state=42), random_forest_parameters))
    models.append(('GBT', GradientBoostingRegressor(random_state=42), gbt_parameters))
    models.append(('ANN', MLPRegressor(), ann_parameters))
    models.append(('Lasso', Lasso(), lasso_parameters))
    '''
    models.append(('K-nn', KNeighborsRegressor(), knn_parameters))
    models.append(('LR', LinearRegression(), lr_parameters))


    models.append(('SVR', SVR(), svr_parameters ))
    return (models)


'''
Import data
'''
data=pd.read_csv('Aripiprazol.csv')

#data=data.drop(data.index[[9,2,73]])

data=data[data.k<=9.5]

#SCALE FEATURES
datasets =create_feature_sets(data)





'''
Split on label and predictors
'''

def prepare_data(data):
    Y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    n_samples = X.shape[0]
    m_features = X.shape[1]
    return (X, Y, n_samples, m_features)

scoring = ['neg_mean_squared_error', 'explained_variance', 'neg_mean_absolute_error', 'r2', 'neg_median_absolute_error']


'''
Optimize Models
'''
i=0
results=[]
for feature_set, data in datasets.items():
    X, Y, n_samples, m_features=prepare_data(data)
    models = create_models(n_samples, m_features)
    for name, model, params in models:
        gs = GridSearchCV(model,
                      param_grid=params,
                      scoring=scoring, cv=10,  return_train_score=True, refit=scoring[0], n_jobs=2)
        gs.fit(X, Y)
        print(i+1,feature_set,name)
        result=pd.DataFrame(gs.cv_results_).filter(regex='^(?!split)', axis=1) # create dataframe and filter results from splits
        result = result.filter(regex='^(?!param_)', axis=1)
        vec=result['rank_test_neg_mean_squared_error']==1
        result=result[vec]
        result['Name']=name  # add name of algorithm to log
        result['Feature_Set']=feature_set
        results.append((name, feature_set, result , gs.best_estimator_, {name:gs.best_params_}))




df_predictions=pd.DataFrame(Y) #Creates initial df with ground truth
full_results=[]
best_params_full={}
for name, feature_set, result, best_estimator, best_params in results: #adds
    store_model(best_estimator, name, feature_set)
    full_results.append(result)
    best_params_full.update(best_params)



import json

with open('best_params.json', 'w') as outfile:
    json.dump(best_params_full, outfile)

df_results=pd.concat(full_results)
df_results.to_csv('results.csv')
#df_predictions.to_csv('predictions.csv')

data.k

data.k.iloc[9]
data.k.iloc[4]
data.k.iloc[2]

data[np.abs(data.k-9.0083333329999995)<=0.01]





'''
Run experiments
'''

'''
### SVR



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

