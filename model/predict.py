import joblib
import sqlite3
import os
from datetime import datetime

# ensure database folder exists
os.makedirs("database", exist_ok=True)

# connect to database
conn = sqlite3.connect("database/predictions.db")
cursor = conn.cursor()

# create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_hours REAL,
    predicted_marks REAL,
    prediction_time TEXT
)
""")

# load model
model_path = "model/student_marks_model.pkl"

if not os.path.exists(model_path):
    print("Model file not found. Train the model first.")
    exit()

model = joblib.load(model_path)

try:
    # user input
    hours = float(input("Enter study hours: "))

    if hours < 0:
        raise ValueError("Study hours cannot be negative")

    # prediction
    prediction = model.predict([[hours]])
    marks = float(prediction[0])

    print(f"Predicted Marks: {marks:.2f}")

    # save to database
    cursor.execute(
        "INSERT INTO results (study_hours, predicted_marks, prediction_time) VALUES (?, ?, ?)",
        (hours, marks, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    print("Prediction saved to database.")
except ValueError as e:
    print("Invalid input:", e)
finally:
    conn.close()
