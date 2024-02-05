from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from icare.metrics import harrell_cindex, tAUC, harrell_cindex_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import pandas as pd
from fast_ICARE import cindex_by_feature
import ray
from models import *
from preprocessors import * 


score_metrics  = {
    'C-index': harrell_cindex,
    'tAUC': tAUC
}


def evaluate_and_train_default_model(ml, dataset_name, handle_nan, X_inner, y_inner, cv):
    if handle_nan:
        pipe_default = make_pipeline(
            StandardScaler(),
            ml
        )
    else:
        pipe_default = make_pipeline(
            NanImputer(dataset_name=dataset_name, method='mean'),
            StandardScaler(),
            ml
        )
    try:
        default_score = np.nanmean(cross_val_score(pipe_default, X_inner, y_inner, cv=cv,
                                                   scoring=harrell_cindex_scorer,
                                                   error_score=np.nan))
        pipe_default.fit(X_inner, y_inner)
        return pipe_default, default_score
    except:
        return pipe_default, np.nan


def run_hyperparam_search_and_train(ml, n_search,
                                    X_inner, y_inner, cv,
                                    dataset_name, model_name, param_grid):
    # prepare the necessary for the hyperparameter search
    precomputed_scores = pd.Series(cindex_by_feature(X_inner, y_inner).numpy(), index=X_inner.columns)
    precomputed_correlations = precompute_correlations(X_inner)
    if 'TCGA_' in dataset_name:
        precomputed_nan = precompute_nan_TCGA(X_inner)
    else:
        precomputed_nan = precompute_nan_other(X_inner)
    pipe = make_pipeline(
        DropNanPreprocessor(nan_rates=precomputed_nan),
        MinScorePreprocessor(scores=precomputed_scores),
        CorrRemovalPreprocessor(corrs=precomputed_correlations),
        NanImputer(dataset_name=dataset_name),
        StandardScaler(),
        ml
    )
    name_model_in_pipe = list(pipe.named_steps.keys())[-1]
    nanimputer__method =  ['mean', 'median', 'most_frequent', 'constant', 'knn']
    if model_name == 'ICARE':
        nanimputer__method += [None]
    distributions = {
        'dropnanpreprocessor__max_rate': np.arange(0., 0.1, 0.01).round(2),
        'minscorepreprocessor__min_score': np.arange(0.5, 0.8, 0.01).round(2),
        'corrremovalpreprocessor__threshold': np.arange(0.5, 1., 0.01).round(2),
        'nanimputer__method': nanimputer__method
    }
    for k in param_grid:
        distributions[name_model_in_pipe+'__'+k] = param_grid[k]
    
    
    # run the search
    warnings.filterwarnings("ignore")
    search = RandomizedSearchCV(pipe, distributions,
                                scoring=harrell_cindex_scorer,
                                n_iter=n_search,
                                error_score=np.nan,
                                verbose=0,
                                refit=True,
                                n_jobs=1,
                                cv=cv,
                                random_state=0)
    search.fit(X_inner, y_inner)
    
    return search, search.best_score_


def prepare_model(model_name, X_inner, y_inner, dataset_name,
                  n_inner_splits, inner_split_test_size, n_search):
    
    cv = ShuffleSplit(n_splits=n_inner_splits, test_size=inner_split_test_size, random_state=0)
    
    ml = model_library[model_name]['fn']()
    param_grid = model_library[model_name]['param_grid']
    handle_nan = model_library[model_name]['handle_nan']
    
    default_model, default_model_score = evaluate_and_train_default_model(ml, dataset_name, handle_nan, X_inner, y_inner, cv)
    
    if 'noprepro' not in model_name:
        optimized_model, optimized_model_score = run_hyperparam_search_and_train(ml, n_search,
                                                                                 X_inner, y_inner, cv,
                                                                                 dataset_name, model_name, param_grid)

        if np.isnan(default_model_score) or optimized_model_score > default_model_score:
            return optimized_model
        else:
            return default_model
    
    return default_model
    

@ray.remote
def worker_test_model_on_dataset(X, y, model_name, 
                                 inner_index, test_index, dataset_name,
                                 n_inner_splits, inner_split_test_size, n_search):
    
    # split the dataset
    X_inner, X_test = X.iloc[inner_index], X.iloc[test_index]
    y_inner, y_test = y[inner_index], y[test_index]
    
    
    # train and optimize the model
    fitted_model = prepare_model(model_name, X_inner, y_inner, dataset_name,
                                 n_inner_splits, inner_split_test_size, n_search)
    
    # eval on test set
    inner_pred = fitted_model.predict(X_inner)
    test_pred = fitted_model.predict(X_test)
    results = {}
    for metric in score_metrics:
        try:
            results[metric] = score_metrics[metric](y_true=y_test, y_pred=test_pred)
        except:
            results[metric] = np.nan
        try:
            results[metric+'_train'] = score_metrics[metric](y_true=y_inner, y_pred=inner_pred)
        except:
            results[metric+'_train'] = np.nan
            
    return results


def test_model_on_dataset(model_name, dataset_name, n_workers,
                          n_outter_splits=5, test_size=0.25,
                          n_inner_splits=5, inner_split_test_size=0.25, n_search=10):
    # load dataset
    with open('datasets/'+dataset_name+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
    X, y = data['X'], data['y']
    X_id = ray.put(X)
    
    n_iter = int(n_outter_splits / n_workers)
    results = []
    rs = StratifiedShuffleSplit(n_splits=n_outter_splits, test_size=test_size, random_state=0)
    workers = []
    for inner_index, test_index in rs.split(X, y['event']):
        workers.append(worker_test_model_on_dataset.remote(X_id, y, model_name,
                                                           inner_index, test_index, dataset_name,
                                                           n_inner_splits, inner_split_test_size, n_search))

        if len(workers) >= n_workers:
            results.append(pd.DataFrame(ray.get(workers)))
            workers = []

    if len(workers) > 0:
        results.append(pd.DataFrame(ray.get(workers)))

    results = pd.concat(results).reset_index(drop=True)
    return results
