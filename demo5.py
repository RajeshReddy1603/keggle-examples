import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Lasso
df=pd.read_csv("Mobile Price Prediction.csv")
print(df)
print(df.columns)
print(df.shape)
print(df.info)
x=df.iloc[:,0:5].values
np.set_printoptions(suppress=True)
print(x)
y=df.iloc[:,5].values
print(y)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(Y_test)
model=LinearRegression()
model.fit(X_train,Y_train)
print(model.predict(X_test))
print(model.score(X_test,Y_test)*100)
obj=Lasso(alpha=50,max_iter=200,tol=0.1)
obj.fit(X_train,Y_train)
print(obj.predict(X_test))
print(model.score(X_test,Y_test))
print(df.isna().sum())
final=cross_val_score(obj,x,y,cv=5)
print(final.mean())
print(obj.predict([[6.1,4,64,3000,16]]))


