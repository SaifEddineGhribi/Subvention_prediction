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

class wflow(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'subventions-accordees-et-refusees.csv']):
        super(wflow, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = wflow()

# define the score (specific score for the FAN problem)
class model_metric(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='full model score'):
        self.name = name

    def __call__(self, y_pred, y_test):
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        def classification_metric(y_pred, y_test):
            return f1_score(y_pred, y_test, average="weighted")

        def regression_metric(y_pred, y_test, tol=.05):
            diff = np.abs(y_pred - y_test)
            mean = np.abs(y_pred + y_test) / 2
            ratio = diff / mean
            mask = ratio < tol
            ratio[mask] = 0

            return np.mean(ratio)

        def metric_model(y_pred, y_test):
            y_pred_class, y_pred_reg = y_pred[:len(y_test)], y_pred[len(y_test):]
            idx = np.where
            y_test_class, y_test_reg = y_test[:len(y_test)], y_test[len(y_test):]
            alpha_class, alpha_reg = 0.6, 0.4
            reg_score = regression_metric(y_pred_reg, y_test_reg)
            class_score = classification_metric(y_pred_class, y_test_class)
            return alpha_class * (1-class_score) + alpha_reg*reg_score

        score = metric_model(y_pred, y_test)

        return score

score_types = [
    model_metric(name='full model score'),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X['Numéro de dossier'])

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
