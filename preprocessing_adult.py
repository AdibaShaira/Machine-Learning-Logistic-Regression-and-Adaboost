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

def preprocess_adult():
    data = pd.read_csv("adult.data")
    data.to_csv("adult.csv")
    df = pd.read_csv('adult.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
    'native-country','decision'])
    df["sex"]= pd.get_dummies(df, columns = ["sex"])
    numerical_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
    df['decision']= df['decision'].replace(' >50K', 1)
    df['decision']= df['decision'].replace(' <=50K', 0)
    df['workclass']= df['workclass'].replace(' ?', df['workclass'].value_counts().idxmax())
    df['occupation']= df['occupation'].replace(' ?', df['occupation'].value_counts().idxmax())
    df['native-country']= df['native-country'].replace(' ?', df['native-country'].value_counts().idxmax())
    Label = 'decision'
    train_y=df['decision']
    df=df.drop(['decision'], axis=1)
    for col in df:
       if col not in numerical_features:
         n = len(pd.unique(df[col]))
         df = pd.get_dummies(df, columns = [df[col].name])

    data_test = pd.read_csv("adult.test")
    data_test.to_csv("adult_test.csv")
    df_test = pd.read_csv('adult_test.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
    'native-country','decision'])
    print('shape of test',df_test.shape)
    df["sex"]= df_test["sex"].replace('Female', 1)
    df["sex"]= df_test["sex"].replace('Male', 0)
    numerical_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
    df_test['decision']= df_test['decision'].replace(' >50K', 1)
    df_test['decision']= df_test['decision'].replace(' <=50K', 0)
    df_test['workclass']= df_test['workclass'].replace(' ?', df_test['workclass'].value_counts().idxmax())
    df_test['occupation']= df_test['occupation'].replace(' ?', df_test['occupation'].value_counts().idxmax())
    df_test['native-country']= df_test['native-country'].replace(' ?', df_test['native-country'].value_counts().idxmax())
    Label = 'decision'
    df_test=df_test.drop(['decision'], axis=1)
    print('shape of test',df_test.shape)
   # print("test..",df_test.shape)
    for col in df_test:
       if col not in numerical_features:
         df_test = pd.get_dummies(df_test, columns = [df_test[col].name])
    

    print('shape of test',df_test['sex'])
   # print("test..",df_test.shape)
    

    logreg = LogisticRegression()
    logreg.fit(df, train_y)
    y_pred=logreg.predict(df_test)
    print("Accuracy:",metrics.accuracy_score(test_y,y_pred))
def adult_pre():
    data = pd.read_csv("adult.data")
    data.to_csv("adult.csv")
    df = pd.read_csv('adult.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
    'native-country','decision'])
    numerical_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
    df['decision']= df['decision'].replace(' >50K', 1)
    df['decision']= df['decision'].replace(' <=50K', 0)
    train_y=df['decision']
    df=df.drop(['decision'], axis=1)
    for col in df:
       if col not in numerical_features:
         n = len(pd.unique(df[col]))
         df = pd.get_dummies(df, columns = [df[col].name])

    
    df=df.drop(['native-country_ Yugoslavia'],axis=1)
    print('shape of train..',df.shape)
    train_x=df
    data = pd.read_csv("adult.test")
    data.to_csv("adult_test.csv")
    df = pd.read_csv('adult_test.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
    'native-country','decision'])
    numerical_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
    df['decision']= df['decision'].replace(' >50K.', 1)
    df['decision']= df['decision'].replace(' <=50K.', 0)
    test_y=df['decision']
    print("test y..",test_y)
    df=df.drop(['decision'], axis=1)
    for col in df:
       if col not in numerical_features:
         n = len(pd.unique(df[col]))
         df = pd.get_dummies(df, columns = [df[col].name])
    
    test_x=df
    logreg = LogisticRegression()
    logreg.fit(train_x, train_y)
    y_pred=logreg.predict(test_x)
    print("pred..",y_pred)
    print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
    



adult_pre()
   