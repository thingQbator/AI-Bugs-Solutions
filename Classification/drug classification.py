Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Download the dataset

import requests

def get_data():
    response = requests.get("https://raw.githubusercontent.com/thingQbator/sandbox-ai-datasets/master/drug.csv", "drug.csv")
    with open("drug.csv", "w") as f:
        f.write(response.text)


get_data()

# Import Numpy and Pandas for Vectorized Calculations and Data Wrangling
import pandas as pd
import numpy as np

# Matplotlib for Visualization
import matplotlib.pyplot as plt

# Sklearn for Machine Learning 
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Read the dataset using Pandas
data = pd.read_csv("drug.csv")


# using .head() to look at the first 5 rows. Try .head(n) where n is any integer to view first n rows of the dataset
# Additionaly try .tail() in a similar way
print("First 5 rows of the dataset : \n")
print(data.head())

# .shape tells us that there are 1338 rows and 7 columns in the dataset
print("Shape of the Dataset : ")
print(data.shape)

# .info() gives information on all the columns in the dataset, including number of non null rows, data type of each column
# and total size
print("Dataset Summary : ")
print(data.info())

# .describe() gives a statistical summary of the entire dataset
print("Statistical Summary of the dataset : ")
print(data.describe())

# all the columns that are present in the dataset
print('List of all the columns : ')
print(data.columns)

# Unique values inside Drug column
print("Unique values in Drug column : ")
print(data["Drug"].unique())

# lets see if there are null values in the dataset
print("Checking for any null values : ")
print(data.isnull().sum())


# Separating features and targets
# Keeping aside 33% of data for testing and using the rest for training
X = data.drop(['Drug'], axis=1)
y = data['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# We use an Ordinal Encoder here to handle categorical columns in the dataset
encoder = ce.OrdinalEncoder(cols=['Sex', 'BP', 'Cholesterol'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Creating  DecisionTreeClassifier Object using the gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
# Training the model with Train data
clf_gini.fit(X_train, y_train)


# Predicting on test data using the trained model
y_pred = clf_gini.predict(X_test)
print('Testing-set accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)*100))

# Analyzing why the model made the decision it made
plt.figure(figsize=(12,8))
plt.title("Generated Decision Tree : ")
plt.suptitle("Close this window to continue",fontsize=10)
tree.plot_tree(clf_gini.fit(X_train, y_train)) 
plt.show()
