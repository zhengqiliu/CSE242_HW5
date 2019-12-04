import numpy as np
import pandas as pd
import time
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from graphviz import Source
from IPython.display import SVG
from sklearn import preprocessing
from tabulate import tabulate

data = pd.read_csv("diabetes.csv")
x = data.drop('Outcome', axis=1)
y = data['Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

sizes = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,150,180,200]
res = []
for size in sizes:
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=size)
    clf.fit(x_train, y_train)
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    res.append([size, time.time() - t0,accuracy_score(y_train, y_predict_train), accuracy_score(y_test, y_predict_test)])

print(tabulate(res, headers=['Forest Size', 'Running Time','Accuracy on Training', 'Accuracy on Testing']))

