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

# Create a decision tree classifier
def objective(trial):
    # Define the search space for hyperparameters
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    random_state = trial.suggest_int('random_state', 0, 100)

    # Create the Decision Tree model with the specified hyperparameters
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)

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

# traning and testing
dt = DecisionTreeClassifier(random_state=best_params['random_state'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
# optuna used to find ideal parameter

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()

# Calculate feature importance
coefficients = dt.feature_importances_

# Create a dataframe to store feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})

# Sort the dataframe by importance in descending order
feature_importance = feature_importance.sort_values('Importance', ascending=True)

# Plot the feature importance
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.show()

# coefficients = dt.coef_[0]

# feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True)
# feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
# plt.show()

# getting our measurements
#accuracy and precision
# acc = accuracy_score(y_test, y_pred)
# print("Accurcy = %.3f"% acc)
# precision = precision_score(y_test, y_pred)
# print("Precision = %.3f" % precision)
# #recall and f1
# recall = recall_score(y_test, y_pred)
# print("Recall = %.3f" % recall)
# print('F1 = %.3f' % f1_score(y_test, y_pred))
# #roc auc
# print('ROC_AUC = %.3f' % roc_auc_score(y_test, y_pred))
# #tp, tn, fp, fn
# t_and_f = confusion_matrix(y_test, y_pred)
# conf_table = ConfusionMatrixDisplay(t_and_f)
# conf_table.plot()
# plt.show()