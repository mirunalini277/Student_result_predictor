import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

df = pd.read_csv("data\Student.csv")
print(df.isnull().sum())

print(df.columns)
#no null values so not handling it 

y = df['Result']
x = df.drop(['Result'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

print("Acuuracy", accuracy_score(y_test,y_predict))

joblib.dump(model,".gitignore\model.pkl")
print("Model Saved Successfully.")