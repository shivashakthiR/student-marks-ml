import joblib
import sqlite3

# connect to database
conn = sqlite3.connect("database/predictions.db")
cursor = conn.cursor()

# create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    study_hours REAL,
    predicted_marks REAL
)
""")

# load saved model
model = joblib.load("model/student_marks_model.pkl")

# ask user input
hours = float(input("Enter study hours: "))

# predict
prediction = model.predict([[hours]])
marks = prediction[0]

print("Predicted Marks:", marks)

# save to database
cursor.execute(
    "INSERT INTO results (study_hours, predicted_marks) VALUES (?, ?)",
    (hours, marks)
)

conn.commit()
conn.close()