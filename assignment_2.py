import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
temp = pd.read_csv("temperatures.csv")
print(temp)
temp.head()
temp.dtypes
temp.columns 
temp.describe()
temp.isnull().sum()
print("Print the graph of top values.")
n=int(input("Enter the value: "))
top_n_data = temp.nlargest(n, "ANNUAL")
plt.figure(figsize=(10,10))
plt.title("Top temperature records")
sns.barplot(x=top_n_data.YEAR, y=top_n_data.ANNUAL)

a=temp[["YEAR"]]
b=temp[["JAN"]]
a_train,a_test,b_train,b_test =train_test_split(a,b, test_size=0.25, random_state=0)
len(a_train)
temp.shape
lr = LinearRegression()
print(a_train)
model=lr.fit(a_train, b_train)
r_sq=lr.score(a_train,b_train)
model.intercept_
model.coef_
b_pred=model.predict(a_test)
print(b_pred)
plt.scatter(a_train,b_train, color="cyan") 
plt.plot(a_train, lr.predict(a_train), color="red", linewidth=1)
plt.title("Temp vs Year")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()
plt.scatter(a_test, b_test, color='blue') 
plt.plot(a_test, lr.predict(a_test), color='yellow', linewidth=1)
plt.title("Temp vs Year")
plt.xlabel("Year")
plt.ylabel("Temperature")

plt.show()