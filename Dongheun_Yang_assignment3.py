# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:47:21 2024

@author: USER
"""
#Ex1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_dongheun = pd.read_csv(r"C:\Users\USER\School\Term4\AI\Assignment\Lab Assignment3_Support Vector Machines\breast_cancer.csv")

print(data_dongheun.dtypes)
print(data_dongheun.isnull().sum())
print(data_dongheun.describe())

data_dongheun['bare'].replace('?', np.nan, inplace=True)
data_dongheun['bare'] = data_dongheun['bare'].astype(float)
data_dongheun['bare'].fillna(data_dongheun['bare'].median(), inplace=True)
data_dongheun.drop('ID', axis=1, inplace=True)

data_dongheun.hist(figsize=(10,10))


sns.histplot(data_dongheun['thickness'])  # Example plot
plt.show()
#pair_plot = sns.pairplot(data_dongheun)
#plt.show()

# Separate the features from the class
X = data_dongheun.drop('class', axis=1)
y = data_dongheun['class']

# Assuming a student number ending in 42 for the seed
seed = 42

# Split the data into train 80% and test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Define a function to train SVM classifiers with different kernels and evaluate them
def train_evaluate_svm(kernel_type, C=1.0):
    clf = SVC(kernel=kernel_type, C=C, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
    return accuracy_train, accuracy_test, confusion_matrix_test

# Train and evaluate SVM classifiers with different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
results = {}

for kernel in kernels:
    if kernel == 'linear':
        results[kernel] = train_evaluate_svm(kernel, C=0.1)
    else:
        results[kernel] = train_evaluate_svm(kernel)

print(results)

#Ex2


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib  # For saving the model
data_dongheun = pd.read_csv(r"C:\Users\USER\School\Term4\AI\Assignment\Lab Assignment3_Support Vector Machines\breast_cancer.csv")
data_dongheun['bare'].replace('?', np.nan, inplace=True)
data_dongheun['bare'] = data_dongheun['bare'].astype(float)

# Drop the ID column
data_dongheun.drop(columns='ID', inplace=True)

# Separate features and the class
X = data_dongheun.drop(columns='class')
y = data_dongheun['class']

# Split data into training and testing sets
seed = 42  # Replace with the last two digits of your student number
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Assuming all features are numeric for simplicity
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X.columns)
])

# Combine transformers into a pipeline
num_pipe_dongheun = Pipeline(steps=[('preprocessor', preprocessor)])
num_pipe_dongheun
print(num_pipe_dongheun)

pipe_svm_dongheun = Pipeline(steps=[
    ('num_pipe_dongheun', num_pipe_dongheun),
    ('svc', SVC(random_state=seed))
])

from sklearn.model_selection import GridSearchCV

param_grid = {
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svc__degree': [2, 3]
}

grid_search_dongheun = GridSearchCV(estimator=pipe_svm_dongheun, param_grid=param_grid, scoring='accuracy', refit=True, verbose=3)
grid_search_dongheun.fit(X_train, y_train)
print(grid_search_dongheun)

best_params = grid_search_dongheun.best_params_
print("Best parameters:", best_params)

# Best estimator
best_model_dongheun = grid_search_dongheun.best_estimator_
print("Best estimator:", best_model_dongheun)

# Evaluate on the test set
y_pred = best_model_dongheun.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

# Save the model
joblib.dump(best_model_dongheun, 'best_model_dongheun.joblib')