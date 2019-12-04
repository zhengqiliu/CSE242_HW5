import numpy as np
import pandas as pd
import time
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from graphviz import Source
from IPython.display import SVG
from sklearn import preprocessing
from tabulate import tabulate

data = pd.read_csv("diabetes.csv")
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
data = np.array(data)
x = np.delete(data, -1, 1)
y = data[:, -1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def average_loglikelihood(x,y,theta):
    z = np.dot(x,theta)
    h = sigmoid(z)
    return -loss_function(h, y)

def loss_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def fit(X, y, step_size):
    theta = np.zeros(X.shape[1])
    num_iter = 10000
    old_loss = 1
    cnt = 0

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        loss = loss_function(h, y)
        if abs(old_loss - loss) < 0.0001:
            return theta, cnt
        old_loss = loss
        for i in range(len(y)):
            gradient = np.dot(X[i], (h[i] - y[i]))
        #gradient = np.dot(X.T, (h - y)) / y.size
            theta -= step_size * gradient
            cnt += 1
    return theta, "Can't converge"


def predict_prob(X, theta):
    return sigmoid(np.dot(X, theta))


def predict(X, theta):
    res = []
    for row in X:
        p = np.dot(row, theta)
        if p >= 0:
            res.append(1)
        else:
            res.append(0)
    return res


step_sizes = [5e-7,4e-7,3e-7,2e-7,1e-7, 1e-8, 1e-9]
res = []
for step_size in step_sizes:
    t0 = time.time()
    theta,converge = fit(x_train, y_train, step_size)
    if step_size == 3e-7:
        temp = theta
    y_predict_train = predict(x_train, theta)
    y_predict_test = predict(x_test, theta)
    average_log_likelihood_on_train = average_loglikelihood(x_train, y_train, theta)
    average_log_likelihood_on_test = average_loglikelihood(x_test, y_test, theta)
    res.append([step_size, converge, time.time() - t0, average_log_likelihood_on_train,average_log_likelihood_on_test,1- accuracy_score(y_train, y_predict_train), 1-accuracy_score(y_test, y_predict_test)])

print(tabulate(res, headers=['Step Size', 'Number of iteration', 'Training Time', 'Average Log-likelihood on Train', 'Average Log-Likelihood on Test', 'Error on Train', 'Error on Test']))
print([round(i, 8) for i in list(temp)])