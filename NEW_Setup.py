from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([
    ('normalize', MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('classify', SVR())
])

for featu