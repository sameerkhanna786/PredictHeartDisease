# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("viewDataset.pdf")
import warnings
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier

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

# Fit and score the KNN classifier on varying number of neighbors
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

# Plot the data for easy analysis
plt.plot([k for k in range(1, 21)], knn_scores)
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()
