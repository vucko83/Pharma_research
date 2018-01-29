from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
import pandas as pd
import numpy as np

#http://scikit-learn.org/stable/modules/pipeline.html

pipe=Pipeline([
    ('normalize', )
    ('reduce_dim', )
    ('model')


])
