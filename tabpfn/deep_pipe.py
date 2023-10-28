from deeppipe_api.deeppipe import  DeepPipe

    
import os
import numpy as np
import openml
from evaluate_classifier import auto_ml_dids_test
from meta_dataset_loader import load_OHE_dataset


openml.config.set_cache_directory(os.path.abspath('openml'))
print(openml.config.get_cache_directory())

# Load a specific dataset by its ID

datasets = load_OHE_dataset(auto_ml_dids_test,one_hot_encode=False)
    
rng = np.random.default_rng(seed=42)

results = []

for dataset in datasets:


    # Split the data into training and test sets
    dataset_length = len(dataset['data'])
    
    train_test_split = 512
    
    dataset_indices = np.arange(dataset_length)
    rng.shuffle(dataset_indices)
    
    # dataset['data'] = dataset['data'][dataset_indices]
    # dataset['target'] = dataset['target'][dataset_indices]
    
    # fit_data = dataset['data'][:train_test_split]
    # fit_target = dataset['target'][:train_test_split]
    
    X_train = dataset['data'].iloc[dataset_indices[:train_test_split]]
    y_train = dataset['target'].iloc[dataset_indices[:train_test_split]]
    X_test = dataset['data'].iloc[dataset_indices[train_test_split:]]
    y_test = dataset['target'].iloc[dataset_indices[train_test_split:]]

    deep_pipe = DeepPipe(n_iters = 50,  #bo iterations
                        time_limit = 3600, #in seconds
                        apply_cv = True,
                        create_ensemble = False,
                        ensemble_size = 10,
                        )
    deep_pipe.fit(X_train, y_train)
    test_score = deep_pipe.score(X_test,y_test)
    print("Test acc.:", test_score) 

    # Print the best parameters and score
    results.append(test_score)
    
formatted_data = '	'.join(map(str, results))
print(formatted_data)
    
