import pandas as pd 
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import math
def preprocesstelco():
	data = pd.read_csv("Telco.csv")
	#data = data.sample(frac=1).reset_index(drop=True)
	#print(len(data))
	data = data.dropna(axis = 0)
	data = data.reset_index(drop=True)
	data = data.drop(['customerID'], axis = 1)
	#print(len(data))

	target = data['Churn'] 
	y = target
	x_train,x_test,y_train,y_test = train_test_split(data,y,test_size=.2,random_state = 20)
	x_train = x_train.reset_index(drop=True)
	x_test = x_test.reset_index(drop=True)
    
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    pred = lr.predict(x_test)
    acc=lr.score(x_test,y_test)
    print(acc)
   
def main():
    preprocesstelco()
main()