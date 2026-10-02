import pandas as pd
import joblib


# Load trained model
model = joblib.load("career_model.pkl")

# Example student profile
student = pd.DataFrame([
    {
        "branch": "CSE",
        "programming": 4,
        "analytical_reasoning": 4,
        "hardware": 1,
        "mathematics": 4,
        "communication": 3,
        "data_structures": 4,
        "operating_systems": 3,
        "networking": 2,
        "digital_electronics": 1,
        "machine_learning": 4
    }
])


# Predict career
prediction = model.predict(student)

print("-----------------------------------------")
print("CAREER GUIDANCE RESULT")
print("-----------------------------------------")

print("Predicted Career:", prediction[0])


# Show prediction probabilities
probabilities = model.predict_proba(student)[0]

classes = model.classes_

results = pd.DataFrame({
    "Career": classes,
    "Probability": probabilities
})

results = results.sort_values(
    by="Probability",
    ascending=False
)

print("\nCareer probabilities:")

for _, row in results.head(5).iterrows():
    print(
        f"{row['Career']}: "
        f"{row['Probability'] * 100:.2f}%"
    )