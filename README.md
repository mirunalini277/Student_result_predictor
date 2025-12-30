# Student Result Predictor

A **Machine Learning based Student Result Prediction System** that predicts whether a student will **Pass or Fail** using **Logistic Regression**.

This project demonstrates the complete machine learning workflow including data preprocessing, training, prediction, and saving the trained model.

---

## 📑 Table of Contents

- [Dataset Description](#-dataset-description)
- [Machine Learning Model](#-machine-learning-model)
- [Technologies Used](#-technologies-used)
- [How to Run the Project](#-how-to-run-the-project)
- [Sample Output](#-sample-output)
- [Future Enhancements](#-future-enhancements)

---

## Dataset Description

| Column Name | Description |
|------------|-------------|
| Maths | Marks in Mathematics |
| Physics | Marks in Physics |
| Chemistry | Marks in Chemistry |
| Result | Pass (1) / Fail (0) |

Dataset Source: **Kaggle**

---

## Machine Learning Model

- **Algorithm:** Logistic Regression  
- **Learning Type:** Supervised Learning  
- **Problem Type:** Binary Classification  
- **Evaluation Metric:** Accuracy Score  

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**

---

## How to Run the Project

### Step 1: Install Required Libraries
```bash
pip install pandas numpy scikit-learn joblib
### Step 2: Train the Model
```bash
python model.py
```

✔ Trains the model  
✔ Displays accuracy  
✔ Saves trained model as `model.pkl`

---

### Step 3: Predict Student Result
```bash
python predict.py
```

✔ Loads trained model  
✔ Predicts result  
✔ Saves output in `output/Result_prediction.csv`

---

## Sample Output

| Maths | Physics | Chemistry | Result |
|------|----------|-----------|--------|
| 10 | 98 | 45 | Fail |
| 95 | 96 | 95 | Pass |
| 45 | 90 | 50 | Pass |

---

## Future Enhancements

- Add GUI using Streamlit  
- Improve accuracy with more features  
- Add visualization charts  
- Deploy as a web application  

---
