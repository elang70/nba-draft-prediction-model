import pandas as pd

# Read the data from a CSV file into a DataFrame
df = pd.read_csv('nba_draft_combine_all_years.csv')

# Drop the "hand_with" and "hand_length" columns
df = df.drop(['Hand (Length)', 'Hand (Width)'], axis=1)

# Set draft pick to 1 if drafted, 0 if not drafted
df['Draft pick'] = df['Draft pick'].apply(lambda x: 0 if pd.isna(x) or x == 0 else 1)

# drop rows with missing values
df = df.dropna()

# Count the number of rows with "Draft pick" as 0 and 1
draft_pick_counts = df['Draft pick'].value_counts()
print(draft_pick_counts)

# We see an imbalance in our data, so we will balance it in the ML process


# Convert DataFrame to CSV file
df.to_csv('updated_data.csv', index=False)
