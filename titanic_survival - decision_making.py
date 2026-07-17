#IMPORT LIBRARIES
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#LOAD DATASET
print("\n------ LOAD DATA ------\n")
df = pd.read_csv(r"C:\Users\user\Downloads\Titanic-Dataset.csv")
print(df.head())

#UNDERSTAND THE DATA
print("\n------ DATA INFORMATION ------\n")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nInformation:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

#HANDLE MISSING VALUES
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

#DROP UNNECESSARY COLUMNS
df=df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

#LABEL ENCODING
print("\n------ LABEL ENCODING ------\n")
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=encoder.fit_transform(df[col])
print(df.head())

#FEATURES AND TARGET
X=df.drop("Survived",axis=1)
y=df["Survived"]

#TRAIN TEST SPLIT
print("\n------ TRAIN TEST SPLIT ------\n")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))

#DECISION TREE MODEL
print("\n------ DECISION TREE ------\n")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

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

#FEATURE IMPORTANCE
print("\n------ FEATURE IMPORTANCE ------\n")
importance=pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})
importance=importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance)

#NEW PASSENGER PREDICTION
print("\n------ NEW PASSENGER PREDICTION ------\n")

#Values must be in the same order as X.columns
print("Feature Order:")
print(X.columns)

new_passenger=[[
    3,      #Pclass
    1,      #Sex (Male=1, Female=0)
    25,     #Age
    0,      #SibSp
    0,      #Parch
    7.25,   #Fare
    2       #Embarked (S=2, C=0, Q=1)
]]
prediction=model.predict(new_passenger)

if prediction[0]==1:
    print("Passenger is likely to Survive.")
else:
    print("Passenger is likely to Not Survive.")
model.fit(X_train, y_train)

# TEST MODEL
print("\n------ ACCURACY ------\n")
y_pred=model.predict(X_test)
print("Predicted Values:")
print(y_pred)

# TRAINING ACCURACY
train_pred=model.predict(X_train)
train_accuracy=accuracy_score(y_train,train_pred)
print("\nTraining Accuracy:",train_accuracy)

# TESTING ACCURACY
test_accuracy=accuracy_score(y_test,y_pred)
print("\nTesting Accuracy:",test_accuracy)

# OVERALL ACCURACY
overall_pred=model.predict(X)
overall_accuracy=accuracy_score(y,overall_pred)
print("\nOverall Accuracy:",overall_accuracy)
