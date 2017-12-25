from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import GridSearchCV


'''
Import data
'''
data=pd.read_csv('Aripiprazol.csv')

data.shape
data.head()

'''
Create feature sets
'''


'''
Initialize algorithms
'''

linear_regression=linear_model.LinearRegression()
random_forest=RandomForestRegressor(n_estimators=30, max_depth=3)
# add k-nn, svm, gbt, lasso,


#Parameters, best result by different validation criteria


gs = GridSearchCV(RandomForestRegressor(random_state=42),
                  param_grid={'max_depth': range(2, 5, 1)},
                  scoring=scoring, cv=5, refit='neg_mean_squared_error')

gs.fit(X, Y)
results = gs.cv_results_

pd.DataFrame(results).head()

gs.best_params_
gs.best_estimator_

'''
Set parameters
'''




'''
Set experiments (cross validation)
'''
Y=data.iloc[:,-1]
X=data.iloc[:,:-1]

list(range(2, 403, 10))

'''
Set evaluation measures
'''

scoring = ['explained_variance','neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'neg_mean_squared_log_error', 'neg_median_absolute_error']

'''
Feature selection techniques
'''


'''
Run experiments
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