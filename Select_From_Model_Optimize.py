from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class KVExtractor(TransformerMixin):
    def __init__(self, kvpairs):
        self.kpairs = kvpairs

    def transform(self, X, *_):
        result = []
        for index, rowdata in X.iterrows():
            rowdict = {}
            for kvp in self.kpairs:
                rowdict.update({rowdata[kvp[0]]: rowdata[kvp[1]]})
            result.append(rowdict)
        return result

    def fit(self, *_):
        return self