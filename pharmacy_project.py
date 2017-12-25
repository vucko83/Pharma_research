import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics


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

'''
Define scoring (evaluation measures)
'''

scoring = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'neg_mean_squared_log_error', 'neg_median_absolute_error']


# 6 features

# 15 features

'''
Optimize algorithms
'''

# add k-nn, svm, gbt, lasso, linear regression



gs = GridSearchCV(RandomForestRegressor(random_state=42),
                  param_grid={'max_depth': range(2, 5, 1)},
                  scoring=scoring, cv=5, refit='neg_mean_squared_error', n_jobs=4, )

gs.fit(X, Y)
results = gs.cv_results_


a.columns
'rank_test_neg_mean_squared_log_error'

a.filter(regex='^(?!split)', axis=1)

vec=a['rank_test_neg_mean_squared_log_error']==1
a[vec]



gs.predict(X)

'''
Log results
'''

a=pd.DataFrame(results)
a.columns

a.columns.startswith('split')

a.filter[]

score_vector=['mean_fit_time', 'mean_score_time', 'mean_test_explained_variance',
       'mean_test_neg_mean_absolute_error', 'mean_test_neg_mean_squared_error',
       'mean_test_neg_mean_squared_log_error',
       'mean_test_neg_median_absolute_error', 'mean_test_r2',
       'mean_train_explained_variance', 'mean_train_neg_mean_absolute_error',
       'mean_train_neg_mean_squared_error',
       'mean_train_neg_mean_squared_log_error',
       'mean_train_neg_median_absolute_error', 'mean_train_r2',
       'param_max_depth', 'params', 'rank_test_explained_variance',
       'rank_test_neg_mean_absolute_error', 'rank_test_neg_mean_squared_error',
       'rank_test_neg_mean_squared_log_error',
       'rank_test_neg_median_absolute_error', 'rank_test_r2',
    'std_fit_time', 'std_score_time', 'std_test_explained_variance',
       'std_test_neg_mean_absolute_error', 'std_test_neg_mean_squared_error',
       'std_test_neg_mean_squared_log_error',
       'std_test_neg_median_absolute_error', 'std_test_r2',
       'std_train_explained_variance', 'std_train_neg_mean_absolute_error',
       'std_train_neg_mean_squared_error',
       'std_train_neg_mean_squared_log_error',
       'std_train_neg_median_absolute_error', 'std_train_r2']


pd.DataFrame(results).columns

gs.best_params_
gs.best_estimator_



'''
Set parameters
'''




'''
Set experiments (cross validation)
'''

list(range(2, 403, 10))

'''
Set evaluation measures
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


linear_regression.fit(X, Y)

cv=model_selection.cross_validate(random_forest.fit(X,Y),X,Y,cv=10, scoring=scoring, return_train_score=True)


pd.DataFrame(cv)

print(cv)