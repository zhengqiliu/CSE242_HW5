import numpy as np
import pandas as pd
import time
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from graphviz import Source
from IPython.display import SVG
from sklearn import preprocessing

data = pd.read_csv("diabetes.csv")
x = data.drop('Outcome', axis=1)
y = data['Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
maxDepth = 3
clf = tree.DecisionTreeClassifier(max_depth=maxDepth)
clf = clf.fit(x_train, y_train)

from tabulate import tabulate
res = []
for i in range(2,20):
    maxDepth = i
    t0 = time.time()
    clf = tree.DecisionTreeClassifier(max_depth = maxDepth)
    clf = clf.fit(x_train, y_train)
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    res.append([maxDepth, time.time() - t0, accuracy_score(y_train, y_predict_train), accuracy_score(y_predict_test, y_test)])

print(tabulate(res, headers=['Max Depth', 'Training Time', 'Accuracy on Train', 'Accuracy on Test']))