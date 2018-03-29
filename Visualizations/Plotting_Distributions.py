import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('Aripiprazol.csv')

plt.plot(data['k'])
data.plot(kind='line')
data.k.plot(kind='hist')

plt.hist(np.log(data.k))