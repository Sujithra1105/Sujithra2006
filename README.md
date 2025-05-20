import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Simulate fake transaction data
def generate_data(num=1000):
    data = []
    for _ in range(num):
        amount = round(random.uniform(1, 1000), 2)
        time_gap = random.randint(0, 100000)
        location_change = random.randint(0, 1)
        unusual_vendor = random.randint(0, 1)
        is_fraud = 1 if (amount > 800 and location_change and unusual_vendor) else 0
        data.append([amount, time_gap, location_change, unusual_vendor, is_fraud])
    return pd.DataFrame(data, columns=["Amount", "TimeGap", "LocationChange", "UnusualVendor", "Fraud"])

df = generate_data()

# 2. Split data
X = df.drop("Fraud", axis=1)
y = df["Fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train AI model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
print("Model Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

# 5. Grade a new transaction
def grade_transaction(amount, time_gap, location_change, unusual_vendor):
    features = [[amount, time_gap, location_change, unusual_vendor]]
    prediction = model.predict(features)[0]
    if prediction == 1:
        return "FRAUDULENT - Block or Alert"
    else:
        return "LEGITIMATE - Approve"

# 6. Example grading
print("\nTransaction Check Example:")
example = grade_transaction(950.75, 2000, 1, 1)
print(f"Transaction Result: {example}")
