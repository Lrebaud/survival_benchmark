from sklearn.base import BaseEstimator, TransformerMixin
import torch
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sklearn.ensemble import BaggingRegressor

def torch_nanstd(x): 
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=1).unsqueeze(-1), 2), dim=-1))

def get_comparable_pairs(y_true):
    y = torch.from_numpy(np.ascontiguousarray(y_true['time']))
    e = torch.from_numpy(y_true['event'])

    grid_x, grid_y = torch.meshgrid(y, y)
    grid_x = grid_x.tril()
    grid_y = grid_y.tril()
    diff_truth = grid_x - grid_y

    grid_ex, grid_ey = torch.meshgrid(e, e)
    valid_pairs =((diff_truth < 0) & (grid_ex == 1))
    res1 = torch.stack(torch.where(valid_pairs)).T

    diff_truth = grid_y - grid_x
    valid_pairs =((diff_truth < 0) & (grid_ey == 1))
    res2 = torch.stack(torch.where(valid_pairs)).T.flip(1)

    pairs = torch.cat([res1,res2]).numpy()
    return np.random.permutation(pairs)


def cindex_by_pair(v_high, v_low):
    eval_comparable = (v_high < v_low).float()
    eval_non_comparable = (v_high == v_low).float()
    return (eval_comparable+(eval_non_comparable*0.5))

def cindex_by_feature(X, y):
    device = 'cpu'
    Xtorch = torch.from_numpy(X.values.copy()).to(device)

    pairs = get_comparable_pairs(y)
    pairs = torch.from_numpy(pairs).to(device)
    n_pairs = pairs.shape[0]
    n_patients = y.shape[0]

    features_values_b_low  = Xtorch[pairs[:, 0]]
    features_values_b_high = Xtorch[pairs[:, 1]]    
    features_cindex_by_pair = cindex_by_pair(features_values_b_high, features_values_b_low)
    features_cindex = features_cindex_by_pair.mean(dim=0)
    return features_cindex

class IcareSurvival(BaseEstimator):

    def fit(self, X, y):
        self.emptyC = False
        if X.shape[1] == 0:
            self.emptyC = True
            return
        self.device = 'cpu'
#         self.feature_names = X.columns
        Xtorch = torch.from_numpy(X.copy()).double().to(self.device)
        
        
        pairs = get_comparable_pairs(y)
        pairs = torch.from_numpy(pairs).to(self.device)
        n_pairs = pairs.shape[0]
        n_patients = y.shape[0]
        
        features_values_b_low  = Xtorch[pairs[:, 0]]
        features_values_b_high = Xtorch[pairs[:, 1]]    
        features_cindex_by_pair = cindex_by_pair(features_values_b_high, features_values_b_low)
        features_cindex = features_cindex_by_pair.mean(dim=0)
                
        self.features_signs = (features_cindex > 0.5).float()*2.-1.
        self.features_mean = Xtorch.nanmean(dim=0)
        self.features_std = torch_nanstd(Xtorch.T)
        
        self.mask_features_to_keep = (self.features_std > 0)
        self.features_mean = self.features_mean[self.mask_features_to_keep]
        self.features_std = self.features_std[self.mask_features_to_keep]
        self.features_signs = self.features_signs[self.mask_features_to_keep]
          

    def predict(self, X):
        if self.emptyC:
            return np.ones(len(X))
        Xtorch = torch.from_numpy(X.copy()).double().to(self.device)        
        Xtorch = Xtorch[:, self.mask_features_to_keep]
        Xtorch -= self.features_mean
        Xtorch /= self.features_std
        Xtorch *= self.features_signs
        return Xtorch.nanmean(dim=1).numpy()
    
    
class BaggedIcareSurvival(BaseEstimator):

    def __init__(self, bootstrap_features=False):
        super().__init__()
        self.bootstrap_features = bootstrap_features
        self.model = BaggingRegressor(estimator=IcareSurvival(), 
                                      n_estimators=200,
                                      bootstrap_features=bootstrap_features)

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)