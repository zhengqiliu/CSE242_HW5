import numpy as np
import pandas as pd
import time
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

layer = (8,16,16)
res = []
momentums = np.arange(0.1, 1, 0.1)
alphas = []
i = 1e-5
for _ in range(10):
    alphas.append(i)
    i *= 10

alphas = [0.0001]
cnt = 0
for label in y_test:
    if label == 0:
        cnt += 1
print(cnt/len(y_test))
for moment in momentums:
    for alpha in alphas:
        t0 = time.time()
        clf = MLPClassifier(hidden_layer_sizes=layer, max_iter=10000, solver='sgd', momentum=moment, alpha=alpha)
        clf.fit(x_train, y_train)
        y_predict_train = clf.predict(x_train)
        y_predict_test = clf.predict(x_test)
        res.append([moment, alpha, time.time() - t0, accuracy_score(y_train, y_predict_train), accuracy_score(y_test, y_predict_test)])
        clf = 0

print(tabulate(res, headers=['Momentum','Alpha', 'Running Time', 'Accuracy on Train','Accuracy on Test']))

