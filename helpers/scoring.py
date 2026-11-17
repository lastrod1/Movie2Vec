import numpy as np

def rmse(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(errors ** 2)))

def mae(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(np.abs(errors)))