from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# read data
df = pd.read_csv('nba_draft_combine_all_years.csv')
# drop useless columns
df.drop(['Player', 'Year', 'Unnamed: 0'], axis=1, inplace=True)
# make draft pick column binary
df['Draft pick'] = df['Draft pick'].apply(lambda x: 0 if pd.isna(x) or x == 0 else 1)

# split data into features and target
X = df.drop('Draft pick', axis=1)
y = df['Draft pick']
X.dropna(inplace=True)
y = y[X.index]

# split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train model using balanced class weights, max_iter=5000 for convergence
clf = LogisticRegression(random_state=0, max_iter=5000, class_weight='balanced').fit(x_train, y_train)
# accuracy based on test
print(clf.score(x_test, y_test))
# classification report based on test
pred = clf.predict(x_test)
print(classification_report(y_test, pred))