import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import recordlinkage
import category_encoders as ce
 
class FeatureExtractor(object):
    def __init__(self):
        pass
 
    def fit(self, X_df, y_array):
        def numDoc(X_df):
            # X_df['numDoc'] = X_df['numDoc'][4:]
        preprocessor = ColumnTransformer(
            transformers=[
                ('zipcode', make_pipeline(zipcode_transformer,
                 SimpleImputer(strategy='median')), zipcode_col),
                ('tezip',te,target_cols),
                ('hc',make_pipeline(headCounter,
                SimpleImputer(strategy='median')),hc_cols),
                ('num', numeric_transformer, num_cols),
                ('agee', age_calc, year_col),
                ('date', make_pipeline(date_transformer,
                SimpleImputer(strategy='median')), date_cols),
                ('APE', make_pipeline(APE_transformer,
                SimpleImputer(strategy='median')), APE_col),
                ('merge', make_pipeline(merge_transformer,
                SimpleImputer(strategy='median')), merge_col),
                ('drop cols', 'drop', drop_cols),
                ])
 
 
        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y_array)
        return self
 
    def transform(self, X_df):
        X_array = self.preprocessor.transform(X_df)
        # print(X_array[0:10])
        return X_array
