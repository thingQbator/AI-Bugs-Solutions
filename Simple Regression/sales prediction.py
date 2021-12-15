Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Download the Dataset

import requests

def get_data():
    response = requests.get("https://raw.githubusercontent.com/thingQbator/sandbox-ai-datasets/master/advertising.csv", "advertising.csv")
    with open("advertising.csv", "w") as f:
        f.write(response.text)
get_data()


# Pandas and Numpy for Data Wrangling and Vectorized Calculations
import pandas as pd
import numpy as np
# Matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Sklearn Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Import the dataset using Pandas
data = pd.read_csv("advertising.csv")



# using .head() to look at the first 5 rows. Try .head(n) where n is any integer to view first n rows of the dataset
# Additionaly try .tail() in a similar way
print("First 5 rows of the dataset : ")
print(data.head())


# We realize using .shape that there are 200 rows and 4 columns in the dataset
print("Shape of the data :")
print(data.shape)

# .info() tells us detailed summary 
# on all the columns like - number of non null rows, data types of each of them, their total size etc
print("Dataset Summary : ")
print(data.info())

# .describe() gives us a statistical summary of the dataset
print("Statistical Summary : ")
print(data.describe())

# Lets make sure there are no null values in the dataset
print("Checking for Null Values :")
print(data.isnull().sum())

# lets view the correlation of each column against our target column
# Using scatter plots

plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Checking for Linear Relationship between TV and Sales")
plt.suptitle("Close window to continue",fontsize=10)
plt.scatter(data["TV"],data["Sales"])
plt.show()


plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.title("Checking for Linear Relationship between Newspaper and Sales")
plt.suptitle("Close window to continue",fontsize=10)
plt.scatter(data["Newspaper"],data["Sales"])
plt.show()


plt.xlabel("Radio")
plt.ylabel("Sales")
plt.title("Checking for Linear Relationship between Radio and Sales")
plt.suptitle("Close window to continue",fontsize=10)
plt.scatter(data["Radio"],data["Sales"])
plt.show()


# Plotting the correlation matrix as a heat map
plt.title("Correlation Matrix")
sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
plt.suptitle("Close window to continue",fontsize=10)
plt.show()

# Using only Tv for training as other columns have a very weak correlation with sales
# Reshaping the columns to be 2 dimesnional because, LinearRegression() doesn't accept 1D vectors

X = np.array(data['TV']).reshape(-1,1)
y = np.array(data['Sales']).reshape(-1,1)


# Using 70% of the data for training and the remaining 30% to be used for validation / testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# Creating an Object of LinearRegression()
lr = LinearRegression()

# Training using the train data using .fit() method
lr.fit(X_train,y_train)

# Predicting on test data using the trained model
y_pred = lr.predict(X_test)


# Checking the score using mean squared error
print("Mean Squared Error : ")
print(mean_squared_error(y_pred,y_test))


# Visualizing the predictions using scatter plot
# The red points are the true targets and the green ones are predicted 
plt.scatter(X_test,y_test,c='r',label="True Targets")
plt.scatter(X_test,y_pred,c='g',label='Predicted Targets')
plt.title("Visualizing model performance")
plt.suptitle("Close window to continue",fontsize=10)
plt.legend()
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()
