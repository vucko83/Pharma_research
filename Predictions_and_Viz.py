import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('Aripiprazol.csv')

def read_model(name, feature_set):
    path='Models/'+name+feature_set+'.pkl'
    print(path)
    file=open(path, 'rb')
    model=pkl.load(file)
    return(model)

def create_feature_sets(data):
    predictors_6=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'k']
    # Based on corellation
    predictors_15=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'ShpC', 'H-don', 'H-acc', 'logP', 'pol', 'Ecd', 'Ed', 'Torsion E (Et)', 'Total Energy (E)', 'k']
    return ({'6_features':data[predictors_6], '15_features':data[predictors_15], '30_features':data})

def prepare_data(data):
    Y=data.iloc[:,-1] # label
    X=data.iloc[:,:-1] # input data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return (X, Y)


names=['ANN', 'GBT', 'K-NN', 'Lasso', 'LR', 'Random_Forest']

feature_sets=['_6_features', '_15_features', '_30_features']

datasets=create_feature_sets(data)

df=datasets['15_features']
X,y=prepare_data(df)

df.head()

model=read_model(names[1], feature_sets[1])

y_hat=model.predict(X)
y=np.array(y)


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "regression")))

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

#data['y_hat'] = cross_val_predict(model, X, y, cv=10)

#data['diff']=np.abs(data.y_hat-data.k)

'''

'''

data['y_hat']=cross_val_predict(model, pd.DataFrame(X), y, cv=10)
data['diff']=abs(data.y_hat-data.k)
data['diff'].nlargest(3)


diff=np.column_stack((y,y_hat))
abs_diff=np.abs(diff[:,0]-diff[:,1])
np.sort (abs_diff)

diff[:,0]-diff[:,1]

abs_diff.arg_sort()[-3:][::-1]

np.argsort(abs_diff)

abs_diff[73]

[2,4,9]

y[9]
y[4]
y[2]

np.concatenate(y,y_hat)

np.average(scores)


from sklearn.ensemble import IsolationForest
# fit the model
rng = np.random.RandomState(42)
clf = IsolationForest(random_state=rng)
clf.fit(X)
y_pred_train = clf.predict(X)

X1=X[y_pred_train==1,:]
y1=y[y_pred_train==1]
len(y1)

fig, ax = plt.subplots()
ax.scatter(y, y_hat, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('')
plt.show()