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
        self.clf.fit(X, y)
        idx = np.where(y > 0)[0]
        y_reg = y[idx]
        X_reg = X[idx, :]
        self.reg.fit(X_reg, y_reg)
    
    def predict_proba(self, X):
        pred1 = self.clf.predict(X)
        idx = np.where(pred1 == 0)[0]
        X_reg = X[idx, :]
        pred2 = self.reg.predict(X_reg)
        return np.concatenate((pred1, pred2), axis=None)
