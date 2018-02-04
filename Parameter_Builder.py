from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA


def create_models(n_samples, m_features):
    random_forest_parameters = rf_parameters(n_samples, m_features)
    gbt_parameters = rf_parameters(n_samples, m_features)
    knn_parameters = {'n_neighbors': range(1, 11, 1)}
    ann_parameters = {'learning_rate': ['constant'], 'learning_rate_init': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
                      'momentum': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], 'max_iter': [50000],
                      'hidden_layer_sizes': nn_size(m_features)}
    lasso_parameters = {'alpha': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], 'max_iter': [50000]}
    lr_parameters = {'normalize': [True]}
    svr_parameters = {'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}


'''
Feature selectors

'''

def rf_parameters(n_samples, m_features, m_n_ratio=1):
    random_forest_parameters = {
        'n_estimators': range(10, 61, 20),
        'max_features': n_features_range(n_samples, m_features),
        'min_samples_leaf': np.arange(0.01, 0.06, 0.01),
        'max_depth': range(2, 11, 2),
    }

    return (random_forest_parameters)





def rf_gbt_param_dict(name='classify', estimators=[RandomForestRegressor(), GradientBoostingRegressor()], n_samples=100, m_features=15):
    dict={
        name: estimators,
        name + '__' + 'n_estimators': range(10, 101, 10),
        name + '__' + 'max_features': n_features_range(n_samples, m_features),
        name + '__' + 'min_samples_leaf': np.arange(0.01, 0.06, 0.01),
        name + '__' + 'max_depth': range(2, 11, 2),
    }
    return (dict)


def create_params_PCA_NMF(name='reduce_dim', reducers=[PCA(), NMF()], n_samples=100, m_features=[3, 8, 16, 28] ):

    params=[]
    for m in m_features:
        dict={
            name: reducers,
            name+'__'+'n_components':[m]
        }
        dict.update(rf_gbt_param_dict(m_features=m ))
        params.append(dict)
    return params


a=create_params_PCA_NMF()
a[0]


'''
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(), NMF()],
        'classify': [RandomForestRegressor(), GradientBoostingRegressor()]a
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
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

'''



def n_features_range(n_samples=100, m_features=2, m_n_ratio=1):
    '''

    :param n_samples: 
    :param m_features: 
    :param m_n_ratio: 
    :return: 
    '''

    max_features = int(round(n_samples / m_n_ratio))
    if m_features < max_features:
        max_features = m_features

    if m_features == 1:
        min_features = 1
    else:
        min_features = 2

    step_size = int(round(np.log2(max_features)))

    param_range = list(range(min_features, max_features, step_size))

    if param_range[-1] < max_features:
        param_range.append(max_features)

    return param_range


def rf_parameters(n_samples, m_features, m_n_ratio=1):
    random_forest_parameters = {
        'n_estimators': range(10, 61, 20),
        'max_features': n_features_range(n_samples, m_features),
        'min_samples_leaf': np.arange(0.01, 0.06, 0.01),
        'max_depth': range(2, 11, 2),
    }

    return (random_forest_parameters)


def nn_size(m_features):
    half = int(np.ceil((m_features + 1) / 2))
    quarter = int(np.ceil((m_features + 1) / 4))
    size = [(quarter,), (half,)]
    return (size)
