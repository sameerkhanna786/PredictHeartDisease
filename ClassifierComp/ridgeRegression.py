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
from sklearn.linear_model import Ridge

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

print("\n\nRidgeRegression")
# Fit and score the RR classifier on varying number of alpha
minAlpha = 0
maxAlpha = 10
step = 0.5
alph = minAlpha
lst = []
while alph <= maxAlpha:
	regRR = Ridge(alpha=alph).fit(X_train, y_train)
	lst.append(regRR.score(X_test, y_test))
	alph += step

# Plot the data for easy analysis
plt.plot([str(k/10.0) for k in range(minAlpha, 10*maxAlpha + 1, int(step*10))], lst)
plt.xlabel('Alpha Values')
plt.ylabel('Scores')
plt.title('Ridge Regression Classifier scores for different alpha values')
plt.show()



