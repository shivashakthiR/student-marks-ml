import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# create model folder if not exists
os.makedirs("model", exist_ok=True)

# load dataset
df = pd.read_csv("data/student_marks.csv")

# features and target
X = df[["study_hours"]]
y = df["marks"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# train model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
print("Model Evaluation")
print("----------------")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# save model
model_path = "model/student_marks_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
