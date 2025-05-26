import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load dataset
df = pd.read_csv("dataset.csv")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode string labels ("good", "bad") into integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and label encoder
dump(model, "posture_model.joblib")
dump(encoder, "label_encoder.joblib")

print("✅ Model trained and saved as 'posture_model.joblib'")
