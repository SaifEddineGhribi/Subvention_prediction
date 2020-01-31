from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier()
    
    def fit(self, X, y):
        self.clf.fit(X, y)
    
    def predict_proba(self, X):
        return self.clf.predict(X)