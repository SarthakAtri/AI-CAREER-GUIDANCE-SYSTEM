import os
import sqlite3

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "career_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "career_guidance.db")

# Load trained machine learning model
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
    """Create the database table if it doesn't exist."""
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


# Initialize database
init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read and validate input
    try:
        branch = request.form["branch"]

        skills = {
            field: int(request.form[field])
            for field in SKILL_FIELDS
        }

    except (KeyError, ValueError):
        return "Please fill in all fields correctly.", 400

    # Check rating range
    if any(value < 0 or value > 4 for value in skills.values()):
        return "Ratings must be between 0 and 4.", 400

    # Create student dataframe
    student = pd.DataFrame([
        {
            "branch": branch,
            **skills
        }
    ])

    # Predict career
    prediction = model.predict(student)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(student)[0]

    # Sort careers by probability
    results = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    # Get top 5 career predictions
    top_results = results[:5]

    # Show result page
    return render_template(
        "result.html",
        prediction=prediction,
        results=top_results
    )


@app.route("/feedback", methods=["POST"])
def feedback():

    # Get feedback information
    name = request.form.get("name", "").strip()
    rating = request.form.get("rating", "").strip()
    message = request.form.get("message", "").strip()

    # Validate required fields
    if not rating or not message:
        return "Please provide a rating and feedback.", 400

    # Thank-you page
    return """
    <!DOCTYPE html>
    <html lang="en">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>Thank You</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 80px 20px;
                background: #f8fafc;
            }

            .thank-you {
                max-width: 600px;
                margin: auto;
                background: white;
                padding: 50px 30px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            }

            h1 {
                margin-bottom: 15px;
                font-size: 32px;
            }

            p {
                margin-bottom: 30px;
                color: #555;
                font-size: 17px;
            }

            a {
                display: inline-block;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 8px;
                background: #111;
                color: white;
            }

            a:hover {
                opacity: 0.9;
            }

        </style>

    </head>

    <body>

        <div class="thank-you">

            <h1>
                Thank You! 🎉
            </h1>

            <p>
                Your feedback has been received.
            </p>

            <a href="/">
                Back to CareerGuide
            </a>

        </div>

    </body>

    </html>
    """


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5001)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )