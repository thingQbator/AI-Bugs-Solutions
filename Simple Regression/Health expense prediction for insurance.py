Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Download the Dataset

import requests

def get_data():
    response = requests.get("https://raw.githubusercontent.com/thingQbator/sandbox-ai-datasets/master/insurance.csv", "insurance.csv")
    with open("insurance.csv", "w") as f:
        f.write(response.text)
get_data()


# Import pandas and Numpy for Data wrangling and vectorized calculations

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configuring matplotlib parameters for graph figure size, font size etc

plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.style.use('seaborn-whitegrid')


# Reading the dataset using Pandas
df = pd.read_csv("insurance.csv")

# using .head() to look at the first 5 rows. Try .head(n) where n is any integer to view first n rows of the dataset
# Additionaly try .tail() in a similar way
print("First 5 rows in the dataset :\n")
print(df.head())


# .shape tells us that there are 1338 rows and 7 columns in the dataset
print("Shape of the Dataset")
print(df.shape)

# Lets take a look at a statistical summary of the dataset using .describe()
print("Statistical Summary of the Dataset \n")
print(df.describe())


# Using heatmap to analyse if there are null values in the dataset
plt.figure(figsize=(12,4))
sns.heatmap(df.isnull(),cbar=False)
plt.title("Checking for null values in the dataset")
plt.suptitle("Close this window to continue",fontsize=10)
plt.show()
# we can clearly see that bmi has a few null rows

# Lets fill the null values with the mean value of the same column
df.fillna(df.mean(),inplace=True)


# Lets see if everything is fix
plt.figure(figsize=(12,4))
plt.title("Checking for Null values again after replacing null values with mean")
plt.suptitle("Close this window to continue",fontsize=10)
sns.heatmap(df.isnull(),cbar=False)
plt.show()

# plotting the correlation matrix as a heatmap
corr = df.corr()
plt.title("Correlation Matrix")
sns.heatmap(corr,annot=True)
plt.suptitle("Close this window to continue",fontsize=10)
plt.show()


# Because there are categorical columns in the dataset we need to convert them to numerical
# We are using One Hot Encoding here
categorical_columns = ['sex','children','smoker','region']
df_encode = pd.get_dummies(data=df,prefix='OHE',prefix_sep="_",columns=categorical_columns,drop_first=True,dtype="int8")

# Lets take a look at the newly created columns
print("First 5 rows of the dataset after using one hot encoding : \n")
df_encode.head()

# Again, because the target column is different in scale, the correlations dont go high. 
# Lets use log transform and see if the correlation becomes significant
df_encode['charges'] = np.log(df_encode["charges"])


# lets take a look at the correlation matrix again
corr = df_encode.corr()
plt.title("Correlation matrix after using One Hot Encoding")
plt.suptitle("Close this window to continue",fontsize=10)
sns.heatmap(corr,annot=True)
plt.show()

# Using only 80% of data for training and the remaining 20% for testing
# Using random state to get the same random data you run every time. you can use any integer for random state
# but make sure you use the same integer every time to recreate the same result
from sklearn.model_selection import train_test_split
X = df_encode.drop('charges',axis=1)
y = df_encode["charges"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)


# Model Training using .fit()
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# Predicting on the test data using .predict()
y_pred = lin_reg.predict(X_test)

# Check modep performance using mean_squared_error
print("Mean Squared Error : ")
print(mean_squared_error(y_test,y_pred))

# Visualizing the model performance using scatter plot
# Here, only age has been used, try with other columns to by creating multiple scatter plots
plt.title("Model performance visualization by comparing true against predicted values")
plt.xlabel("Age")
plt.ylabel("charges")
plt.scatter(X_test["age"],y_test,c='r',label='True Targets')
plt.scatter(X_test["age"],y_pred,c='g',label='Predicted Targets')
plt.legend()
plt.suptitle("Close this window to continue",fontsize=10)
plt.show()
