from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
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
from sklearn.feature_selection import f_regression
from sklearn.metrics import make_scorer

#from ReliefF import ReliefF

from skrebate import ReliefF


def n_features_range(n_samples=100, m_features=3, m_n_ratio=1):
    '''
    Used for definition of algorithm parameters based on available number of features
     and samples, after feature selection.

    It is used for definition of grid search params through a pipeline

    :param n_samples: number of samples in the dataframe/matrix
    :param m_features: number of features after feature selection
    :param m_n_ratio: maximal number of features relative to number of samples

    :return: range of numbers of features used for param optimization in algorithms
    '''

    max_features = int(round(n_samples / m_n_ratio))
    if m_features < max_features:
        max_features = m_features

    if m_features == 1:
        min_features = 1
    else:
        min_features = 2

    step_size = int(round(np.log2(max_features)))

    param_range = [a for a in range(min_features, max_features, step_size)]

    if param_range[-1] < max_features:
        param_range.append(max_features)

    return param_range



def nn_size(m_features):
    half = int(np.ceil((m_features + 1) / 2))
    quarter = int(np.ceil((m_features + 1) / 4))
    size = [(quarter,), (half,)]
    return (size)

'''
Feature selectors

'''

def lasso_param_dict(name='classify', estimators=[Lasso()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'alpha': [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        name + '__' + 'max_iter': [100000]
    }
    return (dict)

def SVR_param_dict(name='classify', estimators=[SVR()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'C': [0.01, 0.03, 0.05, 0.1, 0.5, 1, 10, 100, 500, 1000],
        name + '__' + 'kernel': ['linear', 'rbf']
    }
    return (dict)

def rf_param_dict(name='classify', estimators=[RandomForestRegressor()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'n_estimators': range(20, 101, 20),
        name + '__' + 'max_features': n_features_range(n_samples, m_features),
        name + '__' + 'min_samples_leaf': np.arange(0.01, 0.03, 0.05),
        name + '__' + 'max_depth': range(2, 11, 2)
    }
    return (dict)

def gbt_param_dict(name='classify', estimators=[GradientBoostingRegressor()], n_samples=100, m_features=15):
    dict={

        name: estimators,
        name + '__' + 'n_estimators': range(10, 101, 20),
        name + '__' + 'max_features': n_features_range(n_samples, m_features),
        name + '__' + 'min_samples_leaf': np.arange(0.01, 0.03, 0.05),
        name + '__' + 'max_depth': range(2, 11, 2),
        name + '__' + 'learning_rate': [0.01, 0.03, 0.05, 0.1]
    }
    return (dict)

def knn_param_dict(name='classify', estimators=[KNeighborsRegressor()], n_samples=100, m_features=15):
    dict={

        name: estimators,
        name + '__' + 'n_neighbors': range(1,11,1)
    }
    return (dict)


def ann_param_dict(name='classify', estimators=[MLPRegressor()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'learning_rate': ['constant'],
        name + '__' + 'momentum': [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        name + '__' + 'max_iter': [100000],
        name + '__' + 'hidden_layer_sizes': nn_size(m_features)
    }
    return (dict)


def lr_param_dict(name='classify', estimators=[LinearRegression()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'normalize': [False]
    }
    return (dict)


def ridge_param_dict(name='classify', estimators=[Ridge()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'alpha': [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        name + '__' + 'max_iter': [100000]
    }
    return (dict)



algorithms={'Lasso':lasso_param_dict, 'SVR':SVR_param_dict, 'RF':rf_param_dict, 'GBT':gbt_param_dict, 'K_NN':knn_param_dict, 'ANN':ann_param_dict, 'LR':lr_param_dict, 'Ridge':ridge_param_dict}

#algorithms={'RF':rf_param_dict}

#algorithms={'LR':lr_param_dict, 'Lasso':lasso_param_dict }



def create_params_pca_nmf(name='reduce_dim', reducers=[PCA(), NMF()], n_samples=100, m_features=[5, 10, 15, 20, 25, 30], funcs=[]):
    params=[]

    for func in funcs.values():
        for m in m_features:
            dict = {
                name: reducers,
                name+'__'+'n_components':[m]
            }
            dict.update(func(m_features=m))

            params.append(dict.copy())
    return (params)


def lr_feature_scorer(X,y):
    lr=LinearRegression().fit(X,y)
    scores = np.abs(lr.coef_)
    return(scores)


def create_params_k_best(name='reduce_dim', reducers=[SelectKBest(score_func=mutual_info_regression), SelectKBest(score_func=f_regression), SelectKBest(score_func=lr_feature_scorer)], n_samples=100, m_features=[5, 10, 15, 20, 25, 30], funcs=[]):
    params=[]
    for func in funcs.values(): #promenila da bude .values()
        for m in m_features:
            dict = {
                name: reducers,
                name+'__'+'k':[m]
            }
            dict.update(func(m_features=m))

            params.append(dict.copy())
    return (params)



def create_params_Relief(name='reduce_dim', reducers=[ReliefF()], n_samples=100, m_features=[5, 10, 15, 20, 25, 30],  funcs=[]):
    params=[]
    for func in funcs.values(): #promenila da bude .values()
        for m in m_features:
            dict = {
                name: reducers,
                name+'__'+'n_features_to_select':[m],
                name + '__' + 'n_neighbors': [3]
            }
            dict.update(func(m_features=m))

            params.append(dict.copy())
    return (params)






'''
Add other feature selections
'''



#params_dicts_all=create_params_k_best_f(funcs=algorithms)

params_dicts_all=create_params_pca_nmf(funcs=algorithms)+create_params_k_best(funcs=algorithms)+create_params_Relief(funcs=algorithms)

type(params_dicts_all[0])



'''

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)



params_dicts=create_params_pca_nmf(funcs=algorithms)
params_dicts_all=params_dicts+create_params_k_best(funcs=algorithms)



#Create parameters for each feature selection and each algorithm

for name, func in algorithms.items:
    create_params_pca_nmf(name='reduce_dim', reducers=[PCA()], n_samples=100,
                              m_features=[5, 10, 15, 20, 25, 30], funcs=[func])
    create_params_pca_nmf(name='reduce_dim', reducers=[NMF()], n_samples=100,
                          m_features=[5, 10, 15, 20, 25, 30], funcs=[func])
    create_params_k_best(name='reduce_dim', reducers=[SelectKBest(score_func=mutual_info_regression)], n_samples=100,
                         m_features=[5, 10, 15, 20, 25, 30], funcs=[])

'''


'''
def create_params_k_best_f(name='reduce_dim', reducers=[SelectKBest(score_func=f_regression)], n_samples=100, m_features=[5, 10, 15, 20, 25, 30], funcs=[]):
    params=[]
    for func in funcs.values(): #promenila da bude .values()
        for m in m_features:
            dict = {
                name: reducers,
                name+'__'+'k':[m]
            }
            dict.update(func(m_features=m))

            params.append(dict.copy())
    return (params)


def create_params_select_from_model(name='reduce_dim', reducers=[SelectKBest(score_func=lr_feature_scorer)], n_samples=100, m_features=[5, 10, 15, 20, 25, 30], funcs=[]):
    params=[]
    for func in funcs.values(): #promenila da bude .values()
        for m in m_features:
            dict = {
                name: reducers,
                name+'__'+'k':[m]
            }
            dict.update(func(m_features=m))

            params.append(dict.copy())
    return (params)


'''