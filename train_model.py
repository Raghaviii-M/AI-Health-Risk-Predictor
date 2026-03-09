import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Select useful columns
df = df[['Age','Gender','Blood Type','Medical Condition','Admission Type','Billing Amount','Test Results']]

# Encoding
le_gender = LabelEncoder()
le_blood = LabelEncoder()
le_condition = LabelEncoder()
le_admission = LabelEncoder()
le_result = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Blood Type'] = le_blood.fit_transform(df['Blood Type'])
df['Medical Condition'] = le_condition.fit_transform(df['Medical Condition'])
df['Admission Type'] = le_admission.fit_transform(df['Admission Type'])
df['Test Results'] = le_result.fit_transform(df['Test Results'])

X = df.drop("Test Results", axis=1)
y = df["Test Results"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train,y_train)

# accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)

print("Model Accuracy:", accuracy)

# save model
joblib.dump(model,"model.pkl")