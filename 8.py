import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

def replace(s):
    l2=[]
    for i in s:
        if i not in l2:
            l2.append(i)
    for i in range(len(s)):
        pos=l2.index(s[i])
        s[i]=pos
    return s

iris=load_iris()
X=pd.DataFrame(iris.data,columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
print(X.head())
Y=pd.DataFrame(iris.target,columns=['target'])
print(Y.head())

def graph_plot(l,title,s,target):
    plt.subplot(l[0],l[1],l[2])
    if s==1:
        plt.scatter(X.Sepal_Length,X.Sepal_Width,c=colurmap[target])
    else:
        plt.scatter(X.Petal_Length,X.Petal_Width,c=colurmap[target])
    plt.title(title)
    
plt.figure()
colurmap=np.array(['red','lime','green'])
graph_plot([1,2,1],'Sepal',1,Y.target)
graph_plot([1,2,2],'Petal',2,Y.target)
plt.show()

def model_fit(modelname):
    model=modelname(3)
    model.fit(X)
    
    if modelname==KMeans:
        m='Kmeans'
    else:
        m='Em'
        
    y_pred=model.predict(X)
    km=replace(y_pred)
#     print(y_pred)
    
    plt.figure()
    graph_plot([1,2,1],'Real Classification',2,Y.target)
    graph_plot([1,2,2],m,2,y_pred)
    plt.show()
    
#     print(km)
    print("\n Accuarcy:",sm.accuracy_score(Y,km))
    print("\nConfusion Matrix",sm.confusion_matrix(Y,km))
    
model_fit(KMeans)
model_fit(GaussianMixture)
