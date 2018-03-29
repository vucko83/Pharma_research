import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

data=pd.read_csv('Aripiprazol_2.csv')

X=data.iloc[:,:-1]
X.dtypes

a=sns.pairplot(X)
a.show()
