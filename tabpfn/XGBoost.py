import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import os
import openml
from sklearn.model_selection import train_test_split
import numpy as np
from meta_dataset_loader import load_OHE_dataset
from evaluate_classifier import auto_ml_dids_test

openml.config.set_cache_directory(os.path.abspath('openml'))
print(openml.config.get_cache_directory())


datasets = load_OHE_dataset([31],one_hot_encode=False)
    
rng = np.random.default_rng(seed=42)

for dataset in datasets:
            
    dataset_length = len(dataset['data'])
    
    dataset_indices = np.arange(dataset_length)
    rng.shuffle(dataset_indices)
    
    dataset['data'] = dataset['data'][dataset_indices]
    dataset['target'] = dataset['target'][dataset_indices]
    
    fit_data = dataset['data'][:1000]
    fit_target = dataset['target'][:1000]
    clf = xgb.XGBClassifier(tree_method='gpu_hist')
    # Define the parameter grid for cross-validation
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        # Add other parameters as needed
    }
    # Set up GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # Fit the model
    grid_search.fit(fit_data, fit_target)
    # Evaluate the best model on the test data
    best_model = grid_search.best_estimator_
    test_score = best_model.score(dataset['data'][1000:], dataset['target'][1000:])
    print("Test accuracy: {:.2f}".format(test_score))
