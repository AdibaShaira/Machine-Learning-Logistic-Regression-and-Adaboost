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
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import math
class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :		
        self.learning_rate = learning_rate		
        self.iterations = iterations
        self.b = 0
        
    # Function for model training	
    def fit( self, X, Y,ind,weight ) :		
        # no_of_training_examples, no_of_features	
        	
        self.m, self.n = X.shape	
        self.b = 0		
        self.X = X		
        self.Y = Y
        self.trueY=Y
        #print("X....",self.X.shape)
        #print("Y....",self.Y.shape)
        	
        # weight initialization		
        if(ind==0):
            self.W = np.zeros( (self.n, 1) )		
        else:
            self.W=weight
        
        # gradient descent learning
        losses = []
        #print("1...") 
        for i in range( self.iterations ) :			
                self.update_weights()
                #print("2...") 
        
        
    def _loss(h,y):
        return np.mean("formula")
    # Helper function to update weights in gradient descent
    
    def update_weights( self ) :		
       # A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
        #print("X shape...",self.X.shape)

 
        A=np.tanh( self.X.dot( self.W ) + self.b )
        #print("A shape..",A.shape)
        #print("Y shape...",self.Y.shape)
        Z=A
        #loss = np.matmul((self.Y - Z).T, (self.Y - Z))
        # tmp = ( Z - self.Y.T ) 
        # Z_temp=np.dot(Z,Z)
        # one=np.ones(self.m)
        # one=one-Z_temp
        
        #tmp=tmp*one
        #22% ashtese accuracy
        #weight e vul ache ejonne erokom ashtese   
        # one = np.reshape( one, self.m )
        # dW = np.dot( self.X.T,one  ) / self.m  
        
        
        dW = np.matmul(self.X.T, (Z - self.Y)) / self.m     
        #print("dw...",dW.shape)
        
        # db = np.sum( one ) / self.m 
        db = np.sum(Z - self.Y)  / self.m
        # update weights    
        # self.W = self.W - self.learning_rate * dW    
        # self.b = self.b - self.learning_rate * db
        
        # update weights	
        self.W = self.W - self.learning_rate * dW	
        self.b = self.b - self.learning_rate * db
        
        #print("w...",self.W.shape)
    
    
    def predict( self, X) :	
        # self.W=weight
        #print("weight shape...",self.W.shape)	
        Z = np.tanh( X.dot( self.W ) + self.b )		
        #Y = np.where( Z > 0.5, 1, 0 )
        Y = np.where( Z>=0, 1, -1 )	
        
        return Y
    
def adaboost(dataframe,u_dataframe,k,label):
    data_size=dataframe.shape[0]
    w=[]
    h=[]
    z=[]
    x = 1/data_size
    train_y=dataframe[label]
    #print("train_y...",train_y)
    train_x=u_dataframe
    #print("Column...",train_x.columns)
    for i in range(data_size):
        w.append(x)
    for iter in range(k):
        data_copy=dataframe.copy()
        data_copy = data_copy.sample(replace = True, weights = w, frac = 1)
        #print('dataa copy...',data_copy.columns)
        data_Y=data_copy[label]
        data_X=data_copy.drop([label], axis=1)
       
        #print('x copy..',data_X.columns)
        X = data_X.values
        #print('x copy..',data_X.values)
        Y = data_Y.values
        Y=np.reshape(Y, (-1, 1))
        model = LogitRegression( learning_rate = 0.2, iterations = 2 )
        model.fit( X,Y,0,0 )
        error=0.0
        y_pred=model.predict(train_x)
        

       
        for i in range(data_size):
            if train_y[i]!=y_pred[i]:
                error += w[i]
        if  error!=0:
            z_temp = (1.0-error)/error
        else:
            z_temp = float("inf")
        z.append(math.log(z_temp,2))
        if error>0.5:
            print ('Discard')
            continue
        h.append(model)
       # print("len kotoooo h er....",len(h))
        for i in range(data_size):
            if train_y[i] == y_pred[i] and  error!=0:
                w[i] = w[i]*(error/(1.0-error))
        sumofweights = sum(w)
        if sum!=0:
            for i in range(len(w)):
                w[i] = w[i]/sumofweights
        
        else:
            num = len(w)
            wt = 1/num
            for i in range(num):
                w[i] = wt
        
        iter +=1

    return h,z
def adaboost_predict(dataframe,h,z,label):
    predictions=[]    
    train_y=dataframe[label]
    train_x=dataframe.drop([label], axis=1)
    train_y = train_y.values
    k_size=len(h)
    #h er length ashe 5 gg.ar train_y toh 7043!.etar ki hobe?
    #model=LogitRegression(learning_rate = 0.01, iterations = 100)
    for i in range (k_size):
        y_pred=h[i].predict(train_x)
        print("Accuracy in loop:",metrics.accuracy_score(train_y, y_pred))
        predictions.append(y_pred)
    print("predi")
    pred_y_all=[]
    for i in range(len(dataframe)):
        v = 0
        for k in range(k_size):
            v += predictions[k][i]*z[k]
        
        if v>=0:
            pred_y_all.append(1)
        else:
            pred_y_all.append(-1)
    
    print("predy...",len(pred_y_all))
    print("y...",len(train_y))
    correctly_classified=0
    for count in range( np.size( pred_y_all ) ) : 
        if pred_y_all[count] == train_y[count] :			
            correctly_classified = correctly_classified + 1
    print( "Accuracy on test set by our model	 : ", (
    correctly_classified / count ) * 100 )
    



def preprocess_telco():
    df=pd.read_csv("Telco.csv")
    label = 'Churn'
    #target=df['Churn']
    #output er kono null value thakle oitake drop korlam
    df=df.dropna(axis=0)
    df=df.reset_index(drop=True)
    df=df.drop(['customerID'], axis=1)

    not_string =['tenure','TotalCharges','MonthlyCharges']
    
    
    for col in df:
        df[col] = df[col].replace(' ', np.nan)  
        if df[col].isnull().sum()!=0:
            if col not in not_string:
                df[col]=SimpleImputer(strategy='most_frequent')
            else:
                df[col].fillna(method ='pad')
    
    for col in df:
      if col not in not_string:
        n = len(pd.unique(df[col]))
        if n>2:
           df = pd.get_dummies(df, columns = [df[col].name])
    
    df["gender"]= df["gender"].replace('Female', 1)
    df["gender"]= df["gender"].replace('Male', 0)
    df["Partner"]= df["Partner"].replace('Yes', 1)
    df["Partner"]= df["Partner"].replace('No', 0)
    df["Dependents"]= df["Dependents"].replace('Yes', 1)
    df["Dependents"]= df["Dependents"].replace('No', 0)
    df["PhoneService"]= df["PhoneService"].replace('Yes', 1)
    df["PhoneService"]= df["PhoneService"].replace('No', 0)
    df['PaperlessBilling']=df['PaperlessBilling'].replace('Yes',1)
    df['PaperlessBilling']=df['PaperlessBilling'].replace('No',0)
    df["Churn"]= df["Churn"].replace('Yes', 1)
    df["Churn"]= df["Churn"].replace('No',-1)
    
    
    

    #empty string er jaygay nan
    df["TotalCharges"] = df["TotalCharges"].replace({' ' : np.nan})  
    #object to float 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
    #impute
    updated_df = df
    updated_df['TotalCharges']=updated_df['TotalCharges'].fillna(updated_df['TotalCharges'].mean())
    #scalar 
    scaler = StandardScaler()
    col_list=["MonthlyCharges","tenure","TotalCharges"]
    updated_df[col_list]=scaler.fit_transform(updated_df[col_list])
    target=updated_df["Churn"]
    lr=updated_df
    updated_df=updated_df.drop(['Churn'], axis=1)
    y=target
    train_x,test_x,train_y,test_y = train_test_split(updated_df,y,test_size=.2,random_state = 20)
    # logreg = LogisticRegression()
    # logreg.fit(train_x, train_y)
    # y_pred=logreg.predict(test_x)
    # print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
    #our model

    Y = lr["Churn"].values
    # lr = lr.drop(["Churn"], axis=1)
    X = updated_df.values
    Y = np.reshape(Y, (-1, 1))
    
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=.2,random_state = 20)
    model = LogitRegression( learning_rate = 0.2, iterations = 1000 )
    losses=model.fit( train_x, train_y,0,0 )
    # plt.plot(losses)
    # plt.show()	  
    # for i in range(len(losses)):
    #     print(losses[i])
    Y_pred = model.predict( test_x )
    correctly_classified = 0	
    print("pred_y",Y_pred.shape)
    print("test_y",test_y.shape)
    for count in range( np.size( Y_pred ) ) : 
        if test_y[count] == Y_pred[count] :			
            correctly_classified = correctly_classified + 1
    print( "Accuracy on test set by our model	 : ", (
    correctly_classified / count ) * 100 )
    y_temp = np.reshape(test_y, (-1))
    y_temp2 = np.reshape(Y_pred, (-1))
    print(y_temp)
    print(y_temp.shape, y_temp2.shape)
    tn, fp, fn, tp = confusion_matrix(y_temp, y_temp2).ravel()
    print("tp...",tp)
    print("fp...",fp)
    print("fn...",fn)
    print("tn...",tn)
    recall=tp/(tp+fn)
    print("Recall..",recall)
    print("Accuracy..", (tp + tn) / (tp + tn + fp + fn))
    k = 5
    h, z = adaboost(lr,updated_df,k,'Churn')
    print ('Round',k)
    print ('Training Dataset')
    adaboost_predict(lr,h,z,'Churn')
    # for i in range(1,5):
    #     k = 5
    #     h, z = adaboost(lr,updated_df,k,'Churn')
    #     print ('Round',k)
    #     print ('Training Dataset')
    #     adaboost_predict(lr,h,z,'Churn')
        
        
    
        






def preprocess_adult():
    df = pd.read_csv('adult.data', delimiter=", ",names=attributes) 




def main():
    preprocess_telco()
main()