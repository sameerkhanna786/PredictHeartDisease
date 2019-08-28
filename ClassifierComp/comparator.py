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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

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

#load and train KNN
knn_classifier = KNeighborsClassifier(n_neighbors = 8)
knn_classifier.fit(X_train, y_train)
knn = knn_classifier.score(X_test, y_test)

#load and train LogReg
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train,y_train)
lr = lr_classifier.score(X_test, y_test)

#load and train and score RF
rf_classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)
rf_classifier.fit(X_train, y_train)
rf = rf_classifier.score(X_test, y_test)

#load and train and score DT
dt_classifier = DecisionTreeClassifier(max_features = 22, random_state = 0)
dt_classifier.fit(X_train, y_train)
dt = dt_classifier.score(X_test, y_test)

#load and train and score RR
rr_classifier = Ridge(alpha=6.299)
rr_classifier.fit(X_train, y_train)
rr = rr_classifier.score(X_test, y_test)

#load and train and score SVM
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
svc = svc_classifier.score(X_test, y_test)

names = ["KNN", "LogReg", "RandForest", "DecisionTree", "RidgeReg", "SVM"]
scores = [knn, lr, rf, dt, rr, svc]
colors = ["red","gray", "navy","green", "brown", "purple"]

#plot values
plt.bar([i for i in range(0, len(names))], scores, color=colors)
plt.xlabel("Classifiers")
plt.xticks([i for i in range(0, len(names))], names)
plt.ylabel("Accuracy Score")
plt.title("Classifier Comparision")

#label the plot
for i in range(0, len(names)):
	plt.text(i, scores[i]/2, '{:0.2f}'.format(scores[i]), ha='center', va='center', 
	bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.show()
