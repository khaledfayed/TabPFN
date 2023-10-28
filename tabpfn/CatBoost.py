from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

import openml

# Load a specific dataset by its ID
dataset = openml.datasets.get_dataset(31)  # Replace 31 with the desired dataset ID
X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CatBoost classifier
clf = CatBoostClassifier(task_type='GPU' , verbose=0)

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
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test accuracy: {:.2f}".format(test_score))