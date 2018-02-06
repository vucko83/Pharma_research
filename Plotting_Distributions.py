import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('Aripiprazol.csv')

plt.plot(data['k'])
data.plot(kind='line')
data.k.plot(kind='hist')