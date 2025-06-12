import numpy as np

def preprocess_target(y):
    return (y > np.median(y)).astype(int)

def compute_feature_importances(estimators):
    return np.mean([tree.feature_importances_ for tree in estimators], axis=0)
