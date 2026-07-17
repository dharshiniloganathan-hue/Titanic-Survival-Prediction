#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#LOAD DATASET
print("\n------ LOAD DATA ------\n")
df=pd.read_csv(r"C:\Users\user\Downloads\Titanic-Dataset.csv")
print(df.head())

#UNDERSTAND THE DATA
print("\n------- DATA INFORMATION ------\n")
print("Shape of Dataset:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nInformation:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

#DATA PREPROCESSING
print("\n------ DATA PREPROCESSING ------\n")

#Fill missing Age values
df["Age"]=df["Age"].fillna(df["Age"].median())

#Fill missing Embarked values
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

#Drop Cabin column
if "Cabin" in df.columns:
    df.drop("Cabin",axis=1,inplace=True)

#Drop unnecessary columns
drop_columns = ["PassengerId","Name","Ticket"]
for col in drop_columns:
    if col in df.columns:
        df.drop(col,axis=1,inplace=True)
print(df.head())

#LABEL ENCODING
print("\n------ LABEL ENCODING ------\n")
encoder=LabelEncoder()
df["Sex"]=encoder.fit_transform(df["Sex"])
df["Embarked"]=encoder.fit_transform(df["Embarked"])
print(df.head())

#EXPLORATORY DATA ANALYSIS
print("\n------ EDA ------\n")
print(df["Survived"].value_counts())

#Survival Count
plt.figure(figsize=(5,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

#Age Histogram
plt.figure(figsize=(5,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Fare Histogram
plt.figure(figsize=(5,5))
plt.hist(df["Fare"], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

#Correlation Matrix
plt.figure(figsize=(8,6))
plt.imshow(df.corr(), cmap="YlGnBu")
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.title("Correlation Matrix")
plt.show()

#FEATURE SCALING
print("\n------ FEATURE SCALING ------\n")
X=df.drop("Survived", axis=1)
y=df["Survived"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("Scaled Features:")
print(X_scaled[:5])

#TRAIN TEST SPLIT
print("\n------ TRAIN TEST SPLIT ------\n")
X_train, X_test, y_train, y_test=train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))

#MODEL
print("\n------ TRAINING MODEL ------\n")
model=LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model Training Completed.")

#PREDICTION
print("\n------ TEST MODEL ------\n")
y_pred=model.predict(X_test)
print("Predicted Values:")
print(y_pred)

#MODEL EVALUATION
print("\n------ MODEL EVALUATION ------\n")
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#NEW PASSENGER PREDICTION
print("\n------ NEW PREDICTION ------\n")
new_passenger=[[
    3,      #Pclass
    0,      #Sex (0=Male,1=Female)
    25,     #Age
    0,      #SibSp
    0,      #Parch
    8.05,   #Fare
    2       #Embarked
]]
new_scaled=scaler.transform(new_passenger)
prediction=model.predict(new_scaled)
if prediction[0]==1:
    print("Passenger will survive.")
else:
    print("Passenger will not survive.")
   
#Accuracy
print("\nFinal Accuracy:",accuracy)  
