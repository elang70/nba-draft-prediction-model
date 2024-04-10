from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os

# read data
df = pd.read_csv('nba-draft-prediction-model/data/player_data_with_positions.csv')

# drop useless columns
df.drop(['Player', 'Year', 'Unnamed: 0'], axis=1, inplace=True)

# get unique positions
positions = df['Position'].unique()

# loop over each position and train a logistic regression model
for position in positions:
    # filter data for the current position
    position_df = df[df['Position'] == position]
    
    # define features (X) and target variable (y) for the current position
    X = position_df.drop('Draft pick', axis=1)  # features
    y = position_df['Draft pick']  # target variable
    
    # split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # initialize logistic regression model
    logistic_model = LogisticRegression()
    
    # train the model
    logistic_model.fit(X_train, y_train)
    
    # predictions on the testing set
    y_pred = logistic_model.predict(X_test)
    
    # evaluate the model
    print(f"Classification Report for {position}:")
    print(classification_report(y_test, y_pred))