import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import openml
from sklearn.model_selection import train_test_split



dataset = openml.datasets.get_dataset(31)  # Replace 31 with the desired dataset ID
X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the XGBoost classifier
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
grid_search.fit(X_train, y_train)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test accuracy: {:.2f}".format(test_score))
