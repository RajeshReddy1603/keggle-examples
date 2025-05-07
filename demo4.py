import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_csv("new.csv")
print(df)
x=df.drop("price",axis="columns").values
print(x)
y=df.price.values
print(y)
plt.scatter(x,y,color="green")
plt.show()
model=LinearRegression()
model.fit(x,y)
print(model.predict([[5200]]))
plt.scatter(x,y)
plt.plot(x,model.predict(x),color="yellow")
plt.show()