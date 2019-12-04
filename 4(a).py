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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

layers = [10, 30, 100 ,(8,8),(8,16),(8,16,16),(4,16,16),(6,16,16),(10,10),(10,10,10), (10,30,10), (10,20,20,10), (10,100,10),(20,50,20),(100,100,100,100) ,(50,100,100,50)]
res = []
for layer in layers:
    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=layer, max_iter=1000)
    clf.fit(x_train, y_train)
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    res.append([layer, time.time() - t0, accuracy_score(y_train, y_predict_train), accuracy_score(y_test, y_predict_test)])

print(tabulate(res, headers=['Layers', 'Running Time', 'Accuracy on Train', 'Accuracy on Test']))

