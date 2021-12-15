Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Download the dataset

import requests
response = requests.get("https://raw.githubusercontent.com/thingQbator/sandbox-ai-datasets/master/flux-temperature.csv", "flux-temperature.csv")
with open("flux-temperature.csv", "w") as f:
    f.write(response.text)


# Pandas for Data Wrangling and Numpy for vectorized calculations
import pandas as pd
import numpy as np

# Matplotlib and seaborn for Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn - Most popular Machine Learning Library in Python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Read data from the csv file using pandas
data = pd.read_csv("flux-temperature.csv")

# Shape tells us that the data has 39 rows and 2 columns
print("Shape of the Data : ")
print(data.shape)


# using the .head() we can see the first 5 rows in the dataset try .head(n) where n can be any integer to see first n rows.
# try .tail() too 
print("First 5 rows of the dataset : ")
print(data.head())



# To check if there are any null values in the dataset
print("Checking for Null Values : ")
print(data.isnull().sum())

# .corr() to view the correlations among various columns
print("Correlation of Temperature against Heat Flux")
print(data.corr()['Skin Temperature'].sort_values())


# Plotting the correlation matrix as a heatmap for improved interpretability

f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)
plt.suptitle("Close window to continue",fontsize=10)
plt.show()
# Checking if Heat Flux and Skin Temperature have a linear relationship by plotting a scatter plot
sns.scatterplot(data["Heat Flux"],data["Skin Temperature"])
plt.title("Correlation between Features and Targets")
plt.suptitle("Close window to continue",fontsize=10)
plt.show()

# Using only 80% of total data for training and the rest 20% for testing
# Reshaping because LinearRegresssion() accepts only 2D vectors
X = data["Heat Flux"].values.reshape(-1,1)
y = data["Skin Temperature"].values.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)




# Using Standard Scaler to standardize both the columns to one scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)


# Model Training using SKlearn's LinearRegression Class
slr = LinearRegression()
slr.fit(X_train,y_train)

# Prediction on test data using the trained model
y_pred = slr.predict(X_test)



# Validating performance using mean squared error
print("Mean Squared Error : ")
print(mean_squared_error(y_pred,y_test))


# Score of prediction 
print("Prediction score of the model : ")
print(slr.score(X_test,y_test)*100)

# Visualizing the model performance
# Red points are original targets and green ones are predicted
plt.scatter(X_test,y_test,c='r',label='True Targets')
plt.scatter(X_test,y_pred,c='g',label='Predicted Targets')
plt.legend()
plt.title("Model Predictions against True Targets")
plt.suptitle("Close window to continue",fontsize=10)
plt.xlabel("Scaled Flux")
plt.ylabel("Scaled Temperature")
plt.show()
