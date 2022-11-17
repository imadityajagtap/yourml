import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
df = pd.read_csv("Heart.csv") 
print(df)
df.head(303)
shape = df.shape
print(shape)
print(df.notnull())
result = df.isna().sum()
print(result)
result = df.isna().sum().sum()
print(result)
datatypes = df.dtypes
print(datatypes)
c=(df==0).sum(axis=1)
print(c)
print(df['Age'].mean())
new_df = df.filter(['Age','Sex','ChestPain','RestBP','Chol'])
print(new_df)
train,test= train_test_split (df,random_state=0,test_size=0.25)
print(train.shape)
print(test.shape)
actual = list(np.ones(45))+list(np.zeros(55))
print("****************actual array**************")
print(np.array(actual))
predicted = list(np.ones(48))+list(np.zeros(52))
print ("***************pridicted array***************")
print(np.array(predicted))
ConfusionMatrixDisplay.from_predictions(actual,predicted)
print (classification_report(actual,predicted))
