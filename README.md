# survival_benchmark

This is the code used in the article "Similar performance of 8 machine learning models on 71 censored medical datasets: a case for simplicity". 

All the original code used is present in this repository. Code cleaning is on going.


## Data preparation

Thanks to the SurvSet collection, data the preparation can easily be done in one notebook (prepare_datasets.ipynb)

TCGA data is a bit more complicated as it requires the user to download manifests files from the TCGA website, and involves several preparation steps. This process cannot be fully automated. All the code we used is provided in this repo in the TCGA_download directory. However, to help reproduce our results, we provide here a download link to download the TCGA data already formatted and ready to be used to test models.

## Model evaluation

To evaluate one model on all the datasets :  python evaluate_model.py MODEL_NAME
MODEL_NAME should be defined in the "model_library" dictionnary in the "models.py" file.

## Parameters

Evaluation parameters can be adjusted in NCV_params.py.

 * n_workers: how many process to use to evaluate models. THe higher the faster, but also the higher memory consumption
 * n_outter_splits: how many folds to build for the external loop of the nested cross validation (default: 10)
 * n_inner_splits: how many folds to build for the internal loop of the nested cross validation (used for hyperparameter search) (default: 10)
 * n_search: how many trial should the random search do to optimize the models (default: 100)

## Results analysis

All the results in the article can be reproduced with the notebook "results_analysis.ipynb"