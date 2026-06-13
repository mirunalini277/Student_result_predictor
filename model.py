import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/Student.csv")

X = df.drop("Result", axis=1)
y = df["Result"]

# ==========================
# Train-Test Split
# ==========================
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Model Comparison
# ==========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\n===== Model Comparison Results =====\n")

for name, model in models.items():

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ==========================
# Save Best Model
# ==========================
joblib.dump(best_model, "models/best_model.pkl")

print("\n==============================")
print("Best Model :", best_model_name)
print("Best Accuracy :", round(best_accuracy, 4))
print("Model saved successfully")
print("==============================")

# ==========================
# Unseen Data Testing
# ==========================
print("\n===== Testing on Unseen Data =====\n")

unseen_students = pd.DataFrame(
    [
        [85, 80, 78],
        [30, 25, 20],
        [72, 68, 75],
        [45, 50, 55],
        [95, 92, 90],
        [20, 85, 90]
    ],
    columns=["Maths", "Physics", "Chemistry"]
)

predictions = best_model.predict(unseen_students)

for student, prediction in zip(unseen_students.values, predictions):

    result = "PASS" if prediction == 1 else "FAIL"

    print(
        f"Maths={student[0]}, "
        f"Physics={student[1]}, "
        f"Chemistry={student[2]} "
        f"--> {result}"
    )