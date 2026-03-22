import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 
from LogisticRegression import LogisticRegression

ds = datasets.load_breast_cancer()
X,y = ds.data , ds.target
X_train , X_test , y_train, y_test = train_test_split(X,y , test_size = 0.2 , random_state = 1234)

clf = LogisticRegression(lr = 0.01)
clf.fit(X_train , y_train)
y_pred = clf.pred(X_test)

def acc(y_pred , y_test):
    return np.sum(y_pred == y_test)/len(y_test)

accu = acc(y_pred , y_test)
print(accu)


