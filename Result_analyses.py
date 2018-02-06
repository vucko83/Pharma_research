import pandas as pd
df=pd.read_csv('test_results_without_10.csv')
df.shape

df.columns[:20]

df.dtypes

df.param_classify[:5]


df.param_classify.str.split(pat='(')[0]

df['Algorithm']=df.param_classify.map(lambda x: x.split('(')[0])

# Only RMSE
rmse_set=df.groupby(by=df.Algorithm)['mean_test_neg_mean_squared_error'].agg('max')

#Indices for best RMSE
idx = df.groupby(['Algorithm'])['mean_test_neg_mean_squared_error'].transform(max) == df['mean_test_neg_mean_squared_error']
best_rmse=df[idx]

best_rmse.shape



best_rmse.to_csv('results_best_rmse.csv')
best_rmse.columns

without_splits=best_rmse.filter(regex='^(?!split)', axis=1) # create dataframe and filter results from splits

without_splits=without_splits.drop([12,24,36,48], axis=0)

without_splits.to_csv('results_best_rmse_without_10.csv')




df.groupby(by=df.Algorithm)['mean_test_r2'].agg('max')

mean_test_neg_mean_squared_error
