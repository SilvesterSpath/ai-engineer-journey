from sklearn.datasets import load_iris 
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the iris dataset
data = load_iris()

X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset info
print(f"Feature Names: {data.feature_names}")
print(f"Class Names: {data.target_names}")

# Train Random Forest with default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# Define hyperparameters grid
param_grid = {
  'n_estimators': [50, 100, 150],
  'max_depth': [None, 5, 10],
  'min_samples_split': [2, 5, 10],
}

# initialize GridSearchCV
grid_search = GridSearchCV(rf_default, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_best)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Accuracy: {accuracy_grid:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Define hyperparameters grid for Random Forest
param_dist ={
  'n_estimators': np.arange(50, 200, 10),
  'max_depth': [None, 5, 10, 15],
  'min_samples_split': [2, 5, 10, 20],
}

# initialize RandomizedSearchCV
random_search = RandomizedSearchCV(rf_default, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=20, random_state=42)

# Perform the random search
random_search.fit(X_train, y_train)

# Evaluate the best model
best_model = random_search.best_estimator_
y_pred_random = best_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Best Accuracy: {accuracy_random:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_random))


