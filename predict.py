import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

df = pd.DataFrame({
    "Maths": [10, 95, 45],
    "Physics": [98, 96, 90],
    "Chemistry": [45, 95, 50]
})

predictions = model.predict(df)

results = []
for p in predictions:
    results.append("Pass" if p == 1 else "Fail")

output = pd.DataFrame({
    "Maths": df["Maths"],
    "Physics": df["Physics"],
    "Chemistry": df["Chemistry"],
    "Result": results
})

output.to_csv("output/Result_prediction.csv", index=False)
print("Prediction saved successfully")
