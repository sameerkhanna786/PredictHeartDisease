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

dataset = pd.read_csv('heart.csv')

y = dataset['target']
x_data = dataset.drop(["target"],axis=1)
col_names = []
for col in x_data.columns:
	col_names.append(col)

def construct_lst(lst, scale_params, cur_names):
	out = [0]*len(cur_names)

	for i in range(0, len(cur_names)):
		out[i] = (lst[i] - scale_params[cur_names[i]][0])/(scale_params[cur_names[i]][1] - scale_params[cur_names[i]][0])

	return out

def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, x_data = x_data, y=y):
	vals = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
	nonNoneVals = [i for i in vals if i != None]
	if len(nonNoneVals) < 7:
		return -1
	for i in range(0, len(col_names)):
		if vals[i] == None:
			x_data = x_data.drop([col_names[i]],axis=1)
	X = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
	cur_names = []
	for col in x_data.columns:
	        cur_names.append(col)
	scale_params = {}
	for col in cur_names:
		scale_params[col] = [np.min(x_data[col]), np.max(x_data[col])]
	lr = LogisticRegression()
	lr.fit(X,y)
	lst = np.array(construct_lst(nonNoneVals, scale_params, cur_names))
	lst = lst.reshape(1, -1)
	return lr.predict(lst)[0]


#Test cases - quick verification that everything is working properly
"""
print(predict(63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1) == 1)
print(predict(58, 0, 0, 170, 225, 1, 0, 146, 1, 2.8, 1, 2, 1) == 0)
print(predict(63, 1, None, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1) == 1)
print(predict(58, 0, 0, 170, 225, 1, 0, 146, 1, None, 1, 2, 1) == 0)
print(predict(None, 1, None, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1) == 1)
print(predict(58, 0, 0, 170, 225, 1, 0, None, 1, None, 1, 2, 1) == 0)
print(predict(None, 1, None, 145, None, 1, 0, 150, 0, 2.3, 0, 0, 1) == 1)
print(predict(58, 0, 0, 170, 225, 1, 0, None, 1, None, 1, 2, None) == 0)
print(predict(63, None, None, None, None, None, None, None, None, None, None, None, None))
print(predict(1, None, None, None, None, None, None, None, None, None, None, None, None))
""" 

