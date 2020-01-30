from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier()
        self.reg = RandomForestRegressor()
    
    def fit(self, X, y):
        y_class = y.copy()
        y_class[y_class > 0] = 1
        self.clf.fit(X, y_class)
        idx = np.where(y > 0)[0]
        y_reg = y[idx]
        X_reg = X[idx, :]
        self.reg.fit(X_reg, y_reg)
    
    def predict(self, X):
        pred1 = self.clf.predict(X)
        idx = np.where(pred1 == 0)[0]
        X_reg = X[idx, :]
        pred2 = self.reg.predict(X_reg)
        print('classif',np.shape(pred1),np.shape(pred2))
        return np.concatenate((pred1, pred2), axis=None)
