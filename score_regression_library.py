import numpy as np
from sklearn.metrics import make_scorer

def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)


score = make_scorer(my_custom_loss_func, greater_is_better=True)