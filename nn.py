import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

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

clf = MLPClassifier()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

pred = clf.predict(x_test)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.show()