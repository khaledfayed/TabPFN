from scripts.transformer_prediction_interface import TabPFNClassifier
from meta_dataset_loader import load_OHE_dataset
from evaluate_classifier import auto_ml_dids_test
from sklearn.metrics import accuracy_score
import time
import torch
import wandb
import numpy as np

def experiment_1():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tabpfn = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False)
    mettab = TabPFNClassifier(device=device, N_ensemble_configurations=1, only_inference=False, model_string='_step_each_e_best_lr_0.0001')
    
    datasets = load_OHE_dataset(auto_ml_dids_test, one_hot_encode=False, num_augmented_datasets=0, shuffle=False)
    
    rng = np.random.default_rng()

    for dataset in datasets:
            
        dataset_length = len(dataset['data'])
        
        dataset_indices = np.arange(dataset_length)
        rng.shuffle(dataset_indices)
        
        dataset['data'] = dataset['data'][dataset_indices]
        dataset['target'] = dataset['target'][dataset_indices]
    
    print('TabPFN')
    print('='*20, '\n')
    for dataset in datasets:
        
        fit_data = dataset['data'][:512]
        fit_target = dataset['target'][:512]
        tabpfn.fit(fit_data, fit_target)
        y_eval, p_eval = tabpfn.predict(dataset['data'][512:], return_winning_probability=True)
        accuracy = accuracy_score(dataset['target'][512:], y_eval)
        print('Dataset ID:',dataset['id'], 'Accuracy', accuracy, 'Shape:', dataset['data'].shape, 'Labels:', len(np.unique(dataset['target'])))

    print('='*50, '\n')
    print('MetTab')
    print('='*20, '\n')
    
    for dataset in datasets:
            
        fit_data = dataset['data'][:512]
        fit_target = dataset['target'][:512]
        mettab.fit(fit_data, fit_target)
        y_eval, p_eval = mettab.predict(dataset['data'][512:], return_winning_probability=True)
        accuracy = accuracy_score(dataset['target'][512:], y_eval)
        print('Dataset ID:',dataset['id'], 'Accuracy', accuracy, 'Shape:', dataset['data'].shape, 'Labels:', len(np.unique(dataset['target'])))


if __name__ == "__main__":
    experiment_1()