import pandas as pd

data=pd.read_csv('Aripiprazol.csv')

data.columns

predictors_6=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'k']
predictors_15=['Brij', 'pH', 'acetonitril', 'Stretch Bend E',  'Non-1.4 VDW E', 'Rad', 'ShpC', 'H-don', 'H-acc', 'logP', 'pol', 'Ecd', 'Ed', 'Torsion E (Et)', 'Total Energy (E)', 'k']

