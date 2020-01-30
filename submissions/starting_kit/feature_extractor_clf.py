
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):

        print('train', np.shape(X_df), np.shape(y))

        collectivite = 'collectivite'
        annee = 'anneeBudg'
        obj = 'objet du dossier'
        direction = 'direction'
        nature = 'Nature de la subvention'
        beneficiaire = 'beneficiaire'
        secteur = 'secteur activite'
        drop_cols = ["numDoc", "siret"]

        def colect(X):
            return X.values[:, np.newaxis]
        colectivite_transformer = FunctionTransformer(colect, validate=False)

        def objetdoss(X):
            return X.values[:, np.newaxis]
        obj_transformer = FunctionTransformer(objetdoss, validate=False)

        def direct(X):
            return X.values[:, np.newaxis]
        direction_transformer = FunctionTransformer(direct, validate=False)

        def nature_t(X):
            return X.values[:, np.newaxis]
        nature_transformer = FunctionTransformer(nature_t, validate=False)

        def beneficiaire_t(X):
            return X.values[:, np.newaxis]
        beneficiaire_transformer = FunctionTransformer(
            beneficiaire_t, validate=False)

        def secteur_t(X):
           return X.values[:, np.newaxis]
        secteur_transformer = FunctionTransformer(secteur_t, validate=False)
                                                       
        def annee_t(X):
             return X.values[:, np.newaxis]
        annee_transformer = FunctionTransformer(annee_t, validate=False)
       
        preprocessor = ColumnTransformer(transformers=[
                                                      ('col', make_pipeline(colectivite_transformer, OrdinalEncoder(), SimpleImputer(strategy='median')), collectivite),
                                                      ('annee', make_pipeline(annee_transformer, SimpleImputer(strategy='median')), annee),
                                                      ('dir', make_pipeline(direction_transformer, OrdinalEncoder(),SimpleImputer(strategy='median')), direction),
                                                      # ('nature', make_pipeline(nature_transformer, OrdinalEncoder(),SimpleImputer(strategy='median')), nature),
                                                      # ('beneficiaire', make_pipeline(beneficiaire_transformer, OrdinalEncoder(),SimpleImputer(strategy='median')), beneficiaire),
                                                      # ('sect', make_pipeline(secteur_transformer, OrdinalEncoder(),SimpleImputer(strategy='median')), secteur),
                                                      # ('obj', make_pipeline(obj_transformer, OrdinalEncoder() ,SimpleImputer(strategy='median')), obj),
                                                      ('drop cols', 'drop', drop_cols),
                                                      ])
           
        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y)
        return self

    def transform(self, X_df):

      print('transform', np.shape(X_df))
      return self.preprocessor.transform(X_df)