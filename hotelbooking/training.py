import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv(
    r"C:\Users\pc\OneDrive\Desktop\Nest\hotelbooking\data\hotel_bookings.csv"
)

# Only use the features needed by the GUI
features = [
    'lead_time',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'previous_cancellations',
    'adr'
]

# Create X and fill missing values ONLY in these numeric columns
X = df[features].copy()
X = X.fillna(0)

# Target column
y = df['is_canceled']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save model
with open("hotel_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")