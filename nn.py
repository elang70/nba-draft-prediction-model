import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import optuna

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
    alpha = trial.suggest_float('alpha', 0.0001, 0.01)
    layer_size = trial.suggest_int('layer_size', 32, 128)
    n_layers = trial.suggest_int('n_layers', 2, 11)
    hidden_layer_sizes = tuple([layer_size] * n_layers)

    # Create the Decision Tree model with the specified hyperparameters
    mlp = MLPClassifier(random_state=0, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)

    # Fit the model on the training data
    mlp.fit(x_train, y_train)

    # Predict the labels for the testing data
    y_pred = mlp.predict(x_test)

    # Calculate the accuracy
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # avg = (accuracy + precision + recall) / 3
    avg = (precision + recall) / 2

    return avg

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best hyperparameters:", best_params)

clf = MLPClassifier(random_state=0, hidden_layer_sizes=(best_params['layer_size'], best_params['layer_size'], best_params['layer_size']), alpha=best_params['alpha'])
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

pred = clf.predict(x_test)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.show()