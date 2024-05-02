import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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
    neighbors = trial.suggest_int('n_neighbors', 1, 10)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])

    # Create the Decision Tree model with the specified hyperparameters
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)

    # Fit the model on the training data
    knn.fit(x_train, y_train)

    # Predict the labels for the testing data
    y_pred = knn.predict(x_test)

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

clf = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

pred = clf.predict(x_test)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.show()

