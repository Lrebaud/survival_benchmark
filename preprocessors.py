from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def precompute_nan_TCGA(X):
    return (X==0).mean()
    
def precompute_nan_other(X):
    return X.isna().mean()

class DropNanPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, nan_rates, max_rate=1.):
        self.nan_rates = nan_rates
        self.max_rate = max_rate
        
    def fit(self, X, y=None):
        self.selected_features = self.nan_rates[self.nan_rates <= self.max_rate].index.tolist()
        return self

    def transform(self, X):
        return X.loc[:, self.selected_features]
    
    

    
class MinScorePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, scores, min_score=0.5):
        self.min_score = min_score
        self.scores = scores
        self.norm_scores = self.scores.copy()
        self.mask_invert = self.norm_scores < 0.5
        self.norm_scores[self.mask_invert] = 1. - self.norm_scores[self.mask_invert]
        
    def fit(self, X, y=None):
        self.norm_scores_subgroup = self.norm_scores.loc[X.columns]
        self.selected_features = self.norm_scores_subgroup[self.norm_scores_subgroup >= self.min_score].index.tolist()
        return self

    def transform(self, X):
        return X.loc[:, self.selected_features]
    
    

def precompute_correlations(X):
#     corrs = pd.DataFrame(abs(np.corrcoef(X.values, rowvar=False)), columns=X.columns, index=X.columns)    
    corrs = pd.DataFrame(abs(np.ma.corrcoef(np.ma.masked_invalid(X.values), rowvar=False)), columns=X.columns, index=X.columns)
    return corrs

def corrsel_with_threshold(corrs, threshold):
    upper = np.tril(np.full(corrs.shape, np.nan), k=0)+corrs
    nb_corrwith = (upper > threshold).sum()
    to_keep = nb_corrwith[nb_corrwith == 0].index.tolist()
    return to_keep

class CorrRemovalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, corrs, threshold=1.):
        self.threshold = threshold
        self.corrs = corrs
        
    def fit(self, X, y=None):
        if self.threshold == 1.:
            self.selected_features = X.columns.tolist()
            return self
        self.corrs_sub = self.corrs.loc[X.columns, X.columns]
        self.selected_features = corrsel_with_threshold(self.corrs_sub, self.threshold)
        return self

    def transform(self, X):
        return X.loc[:, self.selected_features]
    
    
    
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

class NanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, dataset_name, method=None):
        self.dataset_name = dataset_name
        self.method = method
        
    def fit(self, X, y=None):
        if 'TCGA_' in self.dataset_name:
            return self
        
        if self.method in ['mean', 'median', 'most_frequent', 'constant']:
            self.sk_imputer = SimpleImputer(missing_values=np.nan,
                                            strategy=self.method,
                                            fill_value=-1)
        elif self.method == 'knn':
            self.sk_imputer = KNNImputer(n_neighbors=5)
        
        if self.method != None:
            self.sk_imputer.fit(X)
        
        return self

    def transform(self, X):
        if 'TCGA_' in self.dataset_name or self.method == None:
            return X
        return pd.DataFrame(self.sk_imputer.transform(X), columns=X.columns)