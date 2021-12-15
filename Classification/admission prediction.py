Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Download dataset

import requests
response = requests.get("https://raw.githubusercontent.com/thingQbator/sandbox-ai-datasets/master/admission_predict.csv", "admission_predict.csv")
with open("admission_predict.csv", "w") as f:
    f.write(response.text)


# Import Pandas and Numpy for data wrangling and vectorized calculations
import pandas as pd
import numpy as np

# Seaborn and matplotlib for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

# Load the dataset into a Data Frame
data = pd.read_csv("admission_predict.csv")

# using .head() to look at the first 5 rows. Try .head(n) where n is any integer to view first n rows of the dataset
# Additionaly try .tail() in a similar way
print("First 5 rows of the dataset : ")
print(data.head())

# Serial No.  is not a meaningful column, so lets go ahead and drop it
data.drop("Serial No.",axis=1,inplace=True)

# Separating features and target from the dataset
X = data[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA","Research"]]
y = data["class"]

# Separating features and targets
# Keeping aside 20% of data for testing and using the rest for training
X_train,X_test,y_train,y_test = train_test_split(X,y)

# Creating a LogisticRegression() object and fitting it against training data
lr = LogisticRegression()
lr.fit(X_train,y_train)

# Predict on training data using the trained model
y_pred = lr.predict(X_test)

# Calculating the accuracy score of the model
print("Model Accuracy : {}%".format(accuracy_score(y_test,y_pred)*100))

# Lets check the predictions on a heatmap (Confusion Matrix)
plt.title("Confusion Matrix")
plt.suptitle("Close the window to continue",fontsize=10)
sns.heatmap(confusion_matrix(y_test,y_pred))
plt.show()
