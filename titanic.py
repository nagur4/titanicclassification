import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")


file_path = r"C:\Users\nagur\Downloads\titanic\train.xlsx"

# Load the Titanic dataset from the local Excel file
data = pd.read_excel(file_path)

data.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)
data.dropna(subset=["Embarked", "Age", "Fare"], inplace=True)
data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
data["Embarked"] = LabelEncoder().fit_transform(data["Embarked"])

# Split the data into features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))
