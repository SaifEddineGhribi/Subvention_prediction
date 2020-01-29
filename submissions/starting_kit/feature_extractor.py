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
        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'),
                            compression='zip', low_memory=False)
        # obtain features from award
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = \
            award['Name_processed'].str.replace(r'[^\w]', '')
        award_features = \
            award.groupby(['Name_processed'])['amount'].agg(['count', 'sum'])
 
        def zipcodes(X):
            zipcode_nums = pd.to_numeric(X['Zipcode'], errors='coerce')
            zipcode_nums /= 1000.
            zipcode_nums = np.floor(zipcode_nums)
 
            return zipcode_nums.values[:, np.newaxis]
        zipcode_transformer = FunctionTransformer(zipcodes, validate=False)
 
        #--------------------------------------------------------------------------#
        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])
        #--------------------------------------------------------------------------#
 
        def headCount(X_df):
            # X_df.loc[X_df['Headcount'] > 1000000,'Headcount'] /= 1000
            # X_df.loc[X_df['Headcount'] < 0,'Headcount'] *= -1
            heads = X_df.groupby('Legal_ID')['Headcount'].transform('mean')
            X_df['Headcount'].fillna(heads,inplace=True)
            return X_df[['Headcount']]
 
        headCounter = FunctionTransformer(headCount,validate=False)
        def age(X_df):
            mins = X_df.groupby('Legal_ID').min()['Year']
            maxs = X_df.groupby('Legal_ID').max()['Year']
 
            mins = pd.DataFrame(mins)
            mins = mins.rename(columns={'Year':'minYear'})
 
            maxs = pd.DataFrame(maxs)
            maxs = maxs.rename(columns={'Year':'maxYear'})
        #     print(mins)
            X_dfm = pd.merge(X_df,mins,left_on='Legal_ID',right_on='Legal_ID',how='left')
            X_dfm = pd.merge(X_dfm,maxs,left_on='Legal_ID',right_on='Legal_ID',how='left')
        #     print(X_dfm)
            X_dfm['minYear'] = X_dfm['Year'] - X_dfm['minYear']
            X_dfm['maxYear'] = X_dfm['maxYear'] - X_dfm['minYear']
            return X_dfm[['maxYear']]
 
        age_calc = FunctionTransformer(age,validate=False)
 
        def process_date(X):
            date = pd.to_datetime(X['Fiscal_year_end_date'], format='%Y-%m-%d')
            return np.c_[date.dt.year, date.dt.month]
            # return np.c_[date.dt.month]
 
        date_transformer = FunctionTransformer(process_date, validate=False)
 
        def process_APE(X):
            APE = X['Activity_code (APE)'].str[:2]
            return pd.to_numeric(APE).values[:, np.newaxis]
        APE_transformer = FunctionTransformer(process_APE, validate=False)
 
        def merge_naive(X):
            X['Name'] = X['Name'].str.lower()
            X['Name'] = X['Name'].str.replace(r'[^\w]', '')
            df = pd.merge(X, award_features, left_on='Name',
                          right_on='Name_processed', how='left')
            return df[['count', 'sum']]
        merge_transformer = FunctionTransformer(merge_naive, validate=False)
 
 
 
        num_cols = ['Legal_ID','Headcount','Year']
        zipcode_col = ['Zipcode']
        hc_cols = ['Legal_ID','Headcount']
        date_cols = ['Fiscal_year_end_date']
        APE_col = ['Activity_code (APE)']
        merge_col = ['Name']
        drop_cols = ['Address', 'City']
        year_col = ['Legal_ID','Year']
        target_cols = [
            'Headcount',
            # 'Zipcode',
            'Activity_code (APE)'
            ]
 
        te = ce.target_encoder.TargetEncoder(
                verbose=1,
                cols=target_cols,
                return_df=False)
 
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
        # print(self.preprocessor.get_feature_names())
        return self
 
    def transform(self, X_df):
        X_array = self.preprocessor.transform(X_df)
        # print(X_array[0:10])
        return X_array