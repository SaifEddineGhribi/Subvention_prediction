import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score


problem_title = 'Parisian associations grants prediction challenge'
_target_column_name = 'montant vote'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow

class FAN(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'subventions-accordees-et-refusees.csv']):
        super(FAN, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = FAN()

# define the score (specific score for the FAN problem)
class FAN_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='fan error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_pred_reg, y_test_reg, y_pred_class, y_test_class):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        def classification_metric(y_pred, y_test):
            return f1_score(y_pred, y_test, average="weighted")

        def regression_metric(y_pred, y_test, tol=.05):
            diff = np.abs(y_pred - y_test)
            mean = np.abs(y_pred + y_test) / 2
            ratio = diff / mean
            mask = ratio < tol
            ratio[mask] = 0

            return np.mean(ratio)

        def metric_model(y_pred_reg, y_test_reg, y_pred_class, y_test_class, alpha_class=0.5, alpha_reg=0.5,tol=.05):
            reg_score = regression_metric(y_pred_reg, y_test_reg, tol=tol)
            class_score = classification_metric(y_pred_class, y_test_class)
            return alpha_class * (1 - class_score) + alpha_reg*reg_score

        score = metric_model(y_pred_reg, y_test_reg, y_pred_class, y_test_class, alpha_class=0.5, alpha_reg=0.5)

        return score

score_types = [
    FAN_error(name='fan error', precision=2),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X['Legal_ID'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,
                       compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'subventions-accordees-et-refusees_TRAIN.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'subventions-accordees-et-refusees_TEST.csv'
    return _read_data(path, f_name)