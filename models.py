import scipy
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis, IPCRidge
from sklearn.utils.fixes import loguniform
from fast_ICARE import IcareSurvival, BaggedIcareSurvival
from sksurv.tree import SurvivalTree
import numpy as np
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees, ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis

model_library = {
    'CoxPH': {
        'fn': CoxPHSurvivalAnalysis,
        'param_grid': {
                        'alpha': loguniform(1e-4, 1e4),
#                         'n_iter': scipy.stats.randint(10,1000), 
                       },
        'handle_nan': False,
    },
    'Coxnet': {
        'fn': CoxnetSurvivalAnalysis,
        'param_grid': {
                        'l1_ratio': scipy.stats.uniform(),
                        'alpha_min_ratio': loguniform(1e-4, 1e4),
#                         'max_iter': scipy.stats.randint(10,500), 
                        'normalize': scipy.stats.randint(0,2),
                       },
        'handle_nan': False,
    },

    'Coxnet_noprepro': {
        'fn': CoxnetSurvivalAnalysis,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'ICARE': {
        'fn': IcareSurvival,
        'param_grid': {},
        'handle_nan': True,
    },
    'ICARE_noprepro': {
        'fn': IcareSurvival,
        'param_grid': {},
        'handle_nan': True,
    }, 
    
    'BaggedICARE': {
        'fn': BaggedIcareSurvival,
        'param_grid': {
            'bootstrap_features': [True, False],
        },
        'handle_nan': True,
    }, 
    
    'BaggedICARE_noprepro': {
        'fn': BaggedIcareSurvival,
        'param_grid': {},
        'handle_nan': True,
    }, 
    
    
    'DecisionTree': {
        'fn': SurvivalTree,
        'param_grid': {
                        'splitter': ['best', 'random'],
                        'max_depth': np.arange(1,20).tolist(),
                        'min_samples_split': scipy.stats.uniform(),
                        'min_samples_leaf': scipy.stats.uniform(),
                        'max_features': ['auto', 'sqrt', 'log2', None],
                       },
        'handle_nan': False,
    },
    'DecisionTree_noprepro': {
        'fn': SurvivalTree,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'LinearSVM': {
        'fn': FastSurvivalSVM,
        'param_grid': {
                        'rank_ratio': [1],
                        'alpha': list(2. ** np.arange(-12, 13, 2)), 
                        'optimizer': ["avltree", "direct-count", "PRSVM", "rbtree", "simple"],
                       },
        'handle_nan': False,
    },
    'LinearSVM_noprepro': {
        'fn': FastSurvivalSVM,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'KernelSVM': {
        'fn': FastKernelSurvivalSVM,
        'param_grid': {
                        'rank_ratio': [1],
                        'alpha': list(2. ** np.arange(-12, 13, 2)), 
                        'optimizer': ["avltree", "rbtree"],
                        'kernel': ['poly', 'rbf', 'sigmoid', 'cosine'],
                        'degree': [2, 3, 4, 5],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, None],
                       },
        'handle_nan': False,
    },

    'KernelSVM_noprepro': {
        'fn': FastKernelSurvivalSVM,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'RandomSurvivalForest': {
        'fn': RandomSurvivalForest,
        'param_grid': {
                        'n_estimators': np.arange(1,500).tolist(),
                        'max_depth': np.arange(1,20).tolist(),
                        'min_samples_split': scipy.stats.uniform(),
                        'min_samples_leaf': scipy.stats.uniform(),
                        'max_features': ['auto', 'sqrt', 'log2', None],
                        'bootstrap': [True, False],
                        'n_jobs': [1],
                       },
        'handle_nan': False,
    },
    'RandomSurvivalForest_noprepro': {
        'fn': RandomSurvivalForest,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'ExtraSurvivalTrees': {
        'fn': ExtraSurvivalTrees,
        'param_grid': {
                        'n_estimators': np.arange(1,500).tolist(),
                        'max_depth': np.arange(1,20).tolist(),
                        'min_samples_split': scipy.stats.uniform(),
                        'min_samples_leaf': scipy.stats.uniform(),
                        'max_features': ['auto', 'sqrt', 'log2', None],
                        'bootstrap': [True, False],
                        'n_jobs': [1],
                       },
        'handle_nan': False,
    },
    'ExtraSurvivalTrees_noprepro': {
        'fn': ExtraSurvivalTrees,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'ComponentwiseGradientBoostingSurvivalAnalysis': {
        'fn': ComponentwiseGradientBoostingSurvivalAnalysis,
        'param_grid': {
                        'loss': ['coxph', 'squared', 'ipcwls'],
                        'n_estimators': np.arange(1,500).tolist(),
                        'learning_rate': list(10.0**np.arange(-6, 6, 1)),
                        'subsample': list(np.arange(0.01,1.01,0.01)),
                        'dropout_rate': list(np.arange(0.,1.,0.01)),
                       },
        'handle_nan': False,
    },
    'ComponentwiseGradientBoostingSurvivalAnalysis_noprepro': {
        'fn': ComponentwiseGradientBoostingSurvivalAnalysis,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'GradientBoostingSurvivalAnalysis': {
        'fn': GradientBoostingSurvivalAnalysis,
        'param_grid': {
                        'loss': ['coxph', 'squared', 'ipcwls'],
                        'n_estimators': np.arange(1,500).tolist(),
                        'learning_rate': list(10.0**np.arange(-6, 6, 1)),
                        'subsample': list(np.arange(0.01,1.01,0.01)),
                        'dropout_rate': list(np.arange(0.,1.,0.01)),
                        'criterion': ['friedman_mse', 'mse', 'mae'],
                        'min_samples_split': scipy.stats.uniform(),
                        'min_samples_leaf': scipy.stats.uniform(),
                        'max_depth': np.arange(1,20).tolist(),
                        'max_features': ['auto', 'sqrt', 'log2', None],
                       },
        'handle_nan': False,
    },
    'GradientBoostingSurvivalAnalysis_noprepro': {
        'fn': GradientBoostingSurvivalAnalysis,
        'param_grid': {},
        'handle_nan': False,
    },
    
    'IPCRidge': {
        'fn': IPCRidge,
        'param_grid': {
                        'alpha': loguniform(1e-4, 1e4),
                        'fit_intercept': [True, False],
                        'copy_X': [True],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                       },
        'handle_nan': False,
    },
    
}


# MinlipSurvivalAnalysis