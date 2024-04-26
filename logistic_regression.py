from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
# ******** MATH OPERATIONS ********
import numpy as np

# ******** DATA MANIPULATION ******
import pandas as pd

# ******** DATA VISUALIZATION *****
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set(style='white', context='notebook')
# ********* SVM ALGORITHMS ********
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn import datasets, neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import optuna
import joblib

# read data
df = pd.read_csv('data/nba-players.csv')

# drop useless columns
df.drop(['name'], axis=1, inplace=True)

# split data into features and target
X = df.drop('target_5yrs', axis=1)
y = df['target_5yrs']
X.dropna(inplace=True)
y = y[X.index]

# split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def objective(trial):
    # Define the search space for hyperparameters
    # max_depth = trial.suggest_int('max_depth', 2, 10)
    max_iter = trial.suggest_int('max_iter', 1, 10000)
    random_state = trial.suggest_int('random_state', 0, 100)

    # Create the Decision Tree model with the specified hyperparameters
    dt = LogisticRegression(random_state=random_state, max_iter=max_iter, class_weight='balanced')

    # Fit the model on the training data
    dt.fit(x_train, y_train)

    # Predict the labels for the testing data
    y_pred = dt.predict(x_test)

    # Calculate the accuracy
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    avg = (accuracy + precision + recall) / 3

    return avg

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# train model using balanced class weights, max_iter=5000 for convergence
clf = LogisticRegression(random_state=best_params['random_state'], max_iter=best_params['max_iter'], class_weight='balanced').fit(x_train, y_train)
# accuracy based on test
print(clf.score(x_test, y_test))
# classification report based on test
pred = clf.predict(x_test)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.show()

coefficients = clf.coef_[0]

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.show()