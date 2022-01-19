import numpy as np 
from sklearn import linear_model 
#load data from file
data = np.genfromtxt('iris.csv', delimiter=',',skip_header=True)

#Distribute data into train and test sets
X_train = data[:80,[0,1,2,3]]
Y_train = data[:80,5]

X_test = data[-20:,[0,1,2,3]]
Y_test = data[-20:,5]
#Define the required Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
def fit_implementation(X_train, Y_train, learning_rate=0.0005, max_iteration=1000):
    #Adding a column of 1's so that the first element of each input is always 1
    #It would be multiplied with theta_0 later
    X_train= np.insert(X_train, 0, values=1, axis=1)
    no_attributes = X_train.shape[1]
    
    #Initialize model parameters theta
    theta = np.zeros((no_attributes,1))
    
    #Run number of iterations
    for icount in range(max_iteration):
        #delta is the quantity that will be added with theta during updating theta
        delta = np.zeros((no_attributes,1))
        totalLogLikelihood = 0
        #Check each data point
        for instance, actualOutput in zip(X_train,Y_train):
            instance=instance.reshape(no_attributes,1)
            dotResult = np.dot(theta.T, instance)
            
            predictedValue=sigmoid(dotResult).squeeze()
            #Calculate the derivative value for this data point
            derivativeValue = instance*(actualOutput-predictedValue)
            #Calculate the amount to be added with theta
            delta += learning_rate*derivativeValue

            logLikelihood = actualOutput*np.log(predictedValue)+(1-actualOutput)*np.log(1-predictedValue)
            totalLogLikelihood += logLikelihood
        theta = theta + delta
        
        #After each 100 iteration, print the status
        if icount%100==0:
            print(icount)
            print(totalLogLikelihood)
            print(theta)
    #print(theta.shape)
    
    return theta


parameters = fit_implementation(X_train, Y_train)
def prediction(X_test, Y_test, theta):
    #Adding a column of 1's so that the first element of each input is always 1
    #It would be multiplied with theta_0 later
    X_test= np.insert(X_test, 0, values=1, axis=1)
    no_attributes = X_test.shape[1]
    
    correctCount = 0
    totalCount = 0
    
    #Check each data point
    for instance, actualOutput in zip(X_test,Y_test):
            instance=instance.reshape(no_attributes,1)
            dotResult = np.dot(theta.T, instance)
            #Calculated the probability of belonging to class 1
            predictedValue=sigmoid(dotResult).squeeze()
            
            if predictedValue >= 0.5:
                predictedOutput = 1
            else:
                predictedOutput = 0
            print(predictedValue, actualOutput)
            if predictedOutput == actualOutput:
                correctCount += 1
            totalCount += 1
    print("Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    
prediction(X_test, Y_test, parameters)