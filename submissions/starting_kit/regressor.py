from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn import preprocessing


from sklearn.model_selection import GridSearchCV
class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0,
             importance_type='gain', learning_rate=0.07, max_delta_step=0,
             max_depth=30, min_child_weight=4, missing=None, n_estimators=10,
             n_jobs=-1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=1, subsample=0.7, verbosity=1)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        # print(self.reg.feature_importances_)
        return self.reg.predict(X)