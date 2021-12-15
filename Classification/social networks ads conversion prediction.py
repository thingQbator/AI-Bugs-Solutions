Python 2.7.18 (v2.7.18:8d21aa21f2, Apr 20 2020, 13:19:08) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Download the dataset

import requests
response = requests.get("https://github.com/thingQbator/sandbox-ai-datasets/blob/master/Social_Network_Ads.csv", "Social_Network_Ads.csv")
with open("Social_Network_Ads.csv", "w") as f:
    f.write(response.text)


# Import numpy and Pandas for Data Wrangling and Vectorized Calculations

import numpy as np
import pandas as pd

# Matplotlib and Seaborn for Data Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Sklearn for Machine Learning
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Read the dataset into a Data Frame
dataset = pd.read_csv("Social_Network_Ads.csv")

# using .head() to look at the first 5 rows. Try .head(n) where n is any integer to view first n rows of the dataset
# Additionaly try .tail() in a similar way
print("First 5 rows of the dataset : \n")
print(dataset.head())

# We will only be using Age and Estimated Salary for prediction
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Separating features and targets
# Keeping aside 20% of data for testing and using the rest for training

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Bring both the features to a common scale using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using Support Vecotr Classifier with a linear Kernel
classifer = SVC(kernel="linear",random_state=0)
classifer.fit(X_train,y_train)

# Predicting on Test data using the trained model
y_pred = classifer.predict(X_test)

# Creating a confusion matrix to visualize classification output

cm = confusion_matrix(y_test,y_pred)
plt.title("Confusion Matrix for Predictions")
plt.suptitle("Close window to continue",fontsize=10)
sns.heatmap(cm)
plt.show()

X_set,y_set = X_train,y_train

# Creating a mesh on 2D plane so that we can visualize the Support Vector Classifier
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min(),stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min(),stop=X_set[:,1].max()+1,step=0.01))
plt.title("Hyperplane created by SVC")
plt.suptitle("Close window to continue",fontsize=10)
plt.contourf(X1,X2,classifer.predict(np.array(np.array([X1.ravel(),X2.ravel()])).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
plt.show()

# Lets put on the data points
# Red points in Green zone are incorrect points
# Green points in Red zone are incorrect points
# The rest are correclty calssified

plt.contourf(X1,X2,classifer.predict(np.array(np.array([X1.ravel(),X2.ravel()])).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title("SVM (Training Set)")
plt.suptitle("Close window to continue",fontsize=10)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Finally lets check the accuracy score of the model

print("Accuracy of the model : {}%".format(accuracy_score(y_test,y_pred)*100))
