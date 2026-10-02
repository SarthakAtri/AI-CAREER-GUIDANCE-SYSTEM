import os
import sqlite3

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Absolute paths so it works no matter where the server starts from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "career_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "career_guidance.db")

model = joblib.load(MODEL_PATH)

SKILL_FIELDS = [
    "programming",
    "analytical_reasoning",
    "hardware",
    "mathematics",
    "communication",
    "data_structures",
    "operating_systems",
    "networking",
    "digital_electronics",
    "machine_learning",
]


def init_db():
    """Create the table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch TEXT,
                programming INTEGER,
                analytical_reasoning INTEGER,
                hardware INTEGER,
                mathematics INTEGER,
                communication INTEGER,
                data_structures INTEGER,
                operating_systems INTEGER,
                networking INTEGER,
                digital_electronics INTEGER,
                machine_learning INTEGER,
                predicted_career TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Read and validate input
    try:
        branch = request.form["branch"]
        skills = {f: int(request.form[f]) for f in SKILL_FIELDS}
    except (KeyError, ValueError):
        return "Please fill in all fields correctly.", 400

    if any(v < 0 or v > 4 for v in skills.values()):
        return "Ratings must be between 0 and 4.", 400

    student = pd.DataFrame([{"branch": branch, **skills}])

    # Predict career + top 5 probabilities
    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]

    results = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    top_results = results[:5]

    # Show prediction result
    return render_template(
        "result.html",
        prediction=prediction,
        results=top_results
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)