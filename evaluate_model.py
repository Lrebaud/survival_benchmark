from NCV import *
import time
from NCV_params import NCV_params
from glob import glob
from datetime import datetime
from tqdm import tqdm
import sys

def evaluate_model_on_dataset(model, dataset):
    try:
        start = time.time()
        results = test_model_on_dataset(model, dataset,
                                        n_workers=NCV_params['n_workers'],
                                        n_outter_splits=NCV_params['n_outter_splits'],
                                        n_inner_splits=NCV_params['n_inner_splits'],
                                        n_search=NCV_params['n_search'])
        elapsed_time = time.time()-start
        results['time'] = elapsed_time
        results.to_csv('results/'+model+'__'+dataset+'.csv', index=False)
    except:
        print(model, dataset, 'failed')
    
def evaluate_model(model):
    print(datetime.now(), 'evaluation of', model, 'started')
    dataset_list = glob('datasets/SurvSet*')+glob('datasets/TCGA*')
    dataset_list =  [x.split('/')[-1].split('.pickle')[0] for x in dataset_list]
    datasets_done = [x.split('__')[-1].split('.csv')[0] for x in glob('results/'+model+'*')]
    n_datasets = len(dataset_list)
    dataset_list = list(set(dataset_list)-set(datasets_done))
    
    
    n_done = len(datasets_done)
    for dataset in tqdm(dataset_list, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        print(datetime.now(), ':', model, 'on', dataset, 'started... (', n_done, '/', n_datasets, ')')
        evaluate_model_on_dataset(model, dataset)
        n_done = len(glob('results/'+model+'__*'))
        print(datetime.now(), ':', model, 'on', dataset, 'done! (', n_done, '/', n_datasets, ')')

if __name__ == '__main__':

    ray.shutdown()
    ray.init(log_to_driver=False)
    
    model = sys.argv[1]
    evaluate_model(model)
