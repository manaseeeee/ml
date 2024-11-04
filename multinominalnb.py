import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('creditcard.csv')
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Basic information about the dataset before dropping null values
print("\nBasic Information Before Dropping Null Values:")
print(df.info())

# Step 3: Drop rows with null values (if any)
df = df.dropna()

# Step 4: Basic information after dropping null values
print("\nBasic Information After Dropping Null Values:")
print(df.info())

# Step 5: Summary statistics of the dataset
print("\nSummary Statistics After Dropping Null Values:")
print(df.describe())

# Step 6: Check for missing values
print("\nMissing Values in Each Column After Dropping Null Values:")
print(df.isnull().sum())

# Step 7: Check for unique values in each column
print("\nNumber of Unique Values in Each Column After Dropping Null Values:")
print(df.nunique())

# Step 8: Separate features (X) and target variable (y)
X = df.drop(columns=['Class'])  # Features
y = df['Class']                 # Target variable

# Step 9: Ensure features are suitable for Multinomial Naive Bayes
# Multinomial Naive Bayes expects non-negative integer feature counts.
# You can scale or binarize your features if necessary, but keep in mind that Multinomial NB expects count data.
# For this dataset, we can normalize the features.
X = (X - X.min()) / (X.max() - X.min())  # Normalize features to be between 0 and 1
X = (X * 10).astype(int)  # Scale features to be in integer counts (0 to 10)

# Step 10: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 11: Create and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 12: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 13: Evaluate the model
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
