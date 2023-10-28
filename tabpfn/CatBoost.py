from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import numpy as np
import openml
from evaluate_classifier import auto_ml_dids_test
from meta_dataset_loader import load_OHE_dataset


openml.config.set_cache_directory(os.path.abspath('openml'))
print(openml.config.get_cache_directory())

# Load a specific dataset by its ID

datasets = load_OHE_dataset([31],one_hot_encode=False)
    
rng = np.random.default_rng(seed=42)

results = []

for dataset in datasets:


    # Split the data into training and test sets
    dataset_length = len(dataset['data'])
    
    train_test_split = 512
    
    dataset_indices = np.arange(dataset_length)
    rng.shuffle(dataset_indices)
    
    dataset['data'] = dataset['data'][dataset_indices]
    dataset['target'] = dataset['target'][dataset_indices]
    
    fit_data = dataset['data'][:train_test_split]
    fit_target = dataset['target'][:train_test_split]

    categorical_features_indices = fit_data.select_dtypes(['category']).columns.tolist()


    # Define the CatBoost classifier
    clf = CatBoostClassifier(task_type='CPU' , verbose=0, cat_features=categorical_features_indices)

    # Define the parameter grid for cross-validation
    param_grid = {
        'iterations': [50, 100, 150],
        'depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.001],
        # Add other parameters as needed
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(fit_data, fit_target)

    # Print the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    best_model = grid_search.best_estimator_
    test_score = best_model.score(dataset['data'][train_test_split:], dataset['target'][train_test_split:])
    print("Test accuracy: {:.2f}".format(test_score))
    
