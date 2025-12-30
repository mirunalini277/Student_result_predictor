import joblib
import pandas as pd

model = joblib.load("model.pkl")

df = pd.DataFrame({"Maths":[10,95,45],"Physics":[98,96,90],"Chemistry":[45,95,50]})

y_predict = model.predict(df)
predicted_result=[]
for i in y_predict:
    if(i==0):
        predicted_result.append("Fail")
    else:
        predicted_result.append("Pass")
d = {"Math":df['Maths'],"Physics":df["Physics"],"Chemistry":df["Chemistry"],
    "Result":predicted_result}
df_result = pd.DataFrame(d)
df_result.to_csv("output\Result_prediction.csv",index=False)