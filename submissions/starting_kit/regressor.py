from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor()
    
    def fit(self, X, y):
        self.reg.fit(X, y)
    
    def predict(self, X):
        return self.reg.predict(X)
