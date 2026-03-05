import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "Hours":[1,2,3,4,5,6,7],
    "Marks":[20,25,35,45,50,60,70]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[5]])
print("Predicted marks for 5 hours:", prediction[0])

joblib.dump(model,"marks_model.pkl")