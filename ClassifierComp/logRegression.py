# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.linear_model import LogisticRegression

# Load the csv dataset
dataset = pd.read_csv('heart.csv')

# Create dummy columns for the data with set values (EX: values only in range 1-4)
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Scale each column to appropriate values using StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Target values are the values we are trying to predict
y = dataset['target']
X = dataset.drop(['target'], axis = 1)

# Set aside 20% of the dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 65)

# Print train/test accuracy
print("Logistic Regression")
lr = LogisticRegression()
lr.fit(X_train,y_train)
print("Train accuracy {}".format(lr.score(X_train, y_train)))
print("Test accuracy {}".format(lr.score(X_test, y_test)))
