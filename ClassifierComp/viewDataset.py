# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import matplotlib.backends.backend_pdf
#pdf = matplotlib.backends.backend_pdf.PdfPages("viewDataset.pdf")
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#The following code is simply to allow viewing of the dataset in an easy to use manner.

#load dataset
dataset = pd.read_csv('heart.csv')

#gleam info about the data
print("Info")
dataset.info()

#compile and plot the correlation matrix
print("\n\nCompiling Correlation Analysis...")
rcParams['figure.figsize'] = 20, 14
corr = dataset.corr()
plt.matshow(corr, cmap='coolwarm')
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
#label each square in the correlation matrix plot
for (i, j), z in np.ndenumerate(corr):
    plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.colorbar()
plt.title("Correlation Analysis Matrix", fontsize=30, fontweight="bold")
#pdf.savefig(plt.gcf())

#save and clear figure
plt.savefig("Correlation Analysis Matrix")
print("Saved!")
plt.clf()

#compile histogram graphs for each column
print("\n\nCompiling Histograms...")
for col in dataset.columns: 
	dataset.hist(column=col)
	plt.xlabel(col[0].upper() + col[1:] + ' Values', fontsize=25)
	plt.ylabel('Count', fontsize=25)
	plt.title(col[0].upper() + col[1:] + " Histogram", fontsize=30, fontweight="bold")
	#pdf.savefig(plt.gcf())
	#save and clear figure
	plt.savefig(col[0].upper() + col[1:] + " Histogram")
	plt.clf()
print("Saved!")
#pdf.close()
