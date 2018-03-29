import pandas as pd

data=pd.read_csv('Aripiprazol.csv')

data.columns

# Based on stepwise MLR method
predictors_6=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'k']

# Based on corellation
predictors_15=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'ShpC', 'H-don', 'H-acc', 'logP', 'pol', 'Ecd', 'Ed', 'Torsion E (Et)', 'Total Energy (E)', 'k']

