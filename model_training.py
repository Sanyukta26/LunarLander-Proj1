import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import json

#1: Load the corresponding CSV dataset with the selected feature representation with better results on the previous phase.
df= pd.read_csv("final_engineered_data.csv")

#2: Split the dataset into training and test sets using: train test split (80% training, 20% test)
#inputs
FEATURES =["x_position", "y_position", "x_velocity", "y_velocity",
           "angle", "angular_velocity", "left_leg", "right_leg",
           "magnitude_velocity", "abs_x_disp", "abs_angle"]
X= df[FEATURES]
Y=df["action"] #target

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

#3: Train a DecisionTreeClassifier using default parameters
clf= DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#4: Evaluate the model on the test set and report: accuracy and confusion matrix
y_pred = clf.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.4f}%")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Nothing(0)", "Left(1)", "Main(2)", "Right(3)"]))

#5: save model using joblib
joblib.dump(clf, 'lunarlander_decisiontree.pkl')
with open("features.json", "w") as f: json.dump(FEATURES, f)
print("Features saved")