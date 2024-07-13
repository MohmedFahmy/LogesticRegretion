import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#read dataset
data= pd.read_csv('diabetes.csv')
data.Outcome.value_counts()
x=data[['Glucose','Age']].values
y=data['Outcome'].values

#visualization
plt.figure(figsize=(10,6))
plt.scatter(x[y==0][:,0] , x[y==0][:,1] ,color='b',label='0')
plt.scatter(x[y==1][:,0] , x[y==1][:,1] ,color='r',label='1')
plt.xlabel('Glucose')
plt.ylabel('Age')
plt.legend()

#learining process
class LogisticRegression:
    
    def __init__(self, l_rate=0.001, iterations=1000):  #assign values for hyper-parameters
        self.l_rate = l_rate  #learning rate
        self.iterations = iterations  #number of iterations

    def fit(self, x, y):  #Fit the training data using Gradient Descent
        self.losses = []  # An empty list to store the error in each iteration
        self.theta = np.zeros((1 + x.shape[1]))  #intitalization,,,Array of zeros 
        n = x.shape[0]  #number of training examples 768
        
        for i in range(self.iterations):
            #Step1
            y_pred = self.theta[0] + np.dot(x, self.theta[1:])  # hypothesis h(x)
            z = y_pred
            #Step2
            g_z =  1 / (1 + np.e**(-z))  #map predicted values to probabilities between 0 & 1      
            
            #Step3
            cost = (-y * np.log(g_z) - (1 - y) * np.log(1 - g_z))/ n #cost function
            self.losses.append(cost) #Tracking losses
            
            #Step4
            d_theta1 = (1/n) * np.dot(x.T, (g_z - y)) #Derivatives of theta[1:]
            d_theta0 = (1/n) * np.sum(g_z - y)  #Derivatives of theta[0]
            
            #Step5
            self.theta[1:] = self.theta[1:] - self.l_rate * d_theta1  #upadting values of thetas using Gradient descent
            self.theta[0] = self.theta[0] - self.l_rate * d_theta0  #upadting the value of theta 0 using Gradient descent     
        return self
    
    
    def predict(self, x):  #Predicts the value after the model has been trained.
        y_pred = self.theta[0] + np.dot(x, self.theta[1:]) 
        z = y_pred
        g_z = 1 / (1 + np.e**(-z))
        return [1 if i > 0.5 else 0 for i in g_z] #Threshold  

#features scaling using z-score
def scale(x):
    x_scaled = x - np.mean(x, axis=0)
    x_scaled = x_scaled / np.std(x_scaled, axis=0)
    return x_scaled

#call the function scale()
x_sd= scale(x) 
model = LogisticRegression()
model.fit(x_sd, y)

# print theta 0, 1, 2
print("theta_0= ", model.theta[0])
print("theta_1= ", model.theta[1])
print("theta_2= ", model.theta[2])

y_pred = model.predict(x_sd)

#compute confusion matrix
CM = confusion_matrix(y_pred, y, labels=[1,0])
print('Confusion Matrix is : \n', CM)

TP=CM[0][0]
FP=CM[0][1]
FN=CM[1][0]
TN=CM[1][1]

ACC = (TP+TN)/(TP+TN+FP+FN)
print('Accuracy is : \n', ACC)
print('--------------------------------')