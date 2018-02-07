import pandas as pd
import numpy as np


df=pd.read_csv('new_results_log.csv')
df.shape

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
df









