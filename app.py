import os
import sqlite3

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "career_model.pkl"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "career_guidance.db"
)


# =========================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# SKILL FIELDS
# =========================================================

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


# =========================================================
# SKILL DISPLAY NAMES
# =========================================================

SKILL_DISPLAY_NAMES = {

    "programming":
        "Programming",

    "analytical_reasoning":
        "Analytical Reasoning",

    "hardware":
        "Hardware",

    "mathematics":
        "Mathematics",

    "communication":
        "Communication",

    "data_structures":
        "Data Structures",

    "operating_systems":
        "Operating Systems",

    "networking":
        "Networking",

    "digital_electronics":
        "Digital Electronics",

    "machine_learning":
        "Machine Learning",
}


# =========================================================
# CAREER DETAILS
# =========================================================

CAREER_DETAILS = {

    "AI/ML Specialist": {

        "description":
            "Works with artificial intelligence and machine learning techniques to build intelligent data-driven systems.",

        "skills": [
            "Machine Learning",
            "Programming",
            "Mathematics",
            "Data Structures"
        ],

        "roadmap": [
            "Learn Python and programming fundamentals",
            "Strengthen mathematics and statistics",
            "Learn machine learning concepts",
            "Build AI and machine learning projects",
            "Develop practical experience through internships or projects"
        ]
    },


    "API Specialist": {

        "description":
            "Works with APIs and helps design, develop, integrate and maintain software interfaces.",

        "skills": [
            "Programming",
            "Communication",
            "Operating Systems",
            "Data Structures"
        ],

        "roadmap": [
            "Learn programming fundamentals",
            "Understand HTTP and REST APIs",
            "Learn API development and testing",
            "Build API-based projects",
            "Practice API integration"
        ]
    },


    "Application Support Engineer": {

        "description":
            "Provides technical support for software applications and helps identify and resolve application-related issues.",

        "skills": [
            "Programming",
            "Communication",
            "Operating Systems",
            "Analytical Reasoning"
        ],

        "roadmap": [
            "Learn programming fundamentals",
            "Understand operating systems",
            "Develop troubleshooting skills",
            "Learn application support processes",
            "Gain practical support experience"
        ]
    },


    "Business Analyst": {

        "description":
            "Analyzes business requirements and helps organizations improve processes and make data-informed decisions.",

        "skills": [
            "Analytical Reasoning",
            "Communication",
            "Mathematics",
            "Programming"
        ],

        "roadmap": [
            "Develop analytical thinking",
            "Learn Excel and data analysis",
            "Learn SQL and basic programming",
            "Understand business requirements",
            "Practice with real business case studies"
        ]
    },


    "Customer Service Executive": {

        "description":
            "Supports customers by understanding their problems, providing solutions and maintaining effective communication.",

        "skills": [
            "Communication",
            "Analytical Reasoning",
            "Operating Systems"
        ],

        "roadmap": [
            "Improve communication skills",
            "Develop problem-solving skills",
            "Learn customer support tools",
            "Practice handling customer queries",
            "Gain customer service experience"
        ]
    },


    "Cyber Security Specialist": {

        "description":
            "Works to protect computer systems, networks and information from security threats.",

        "skills": [
            "Networking",
            "Operating Systems",
            "Programming",
            "Analytical Reasoning"
        ],

        "roadmap": [
            "Learn networking fundamentals",
            "Understand operating systems",
            "Learn cybersecurity fundamentals",
            "Practice security tools and labs",
            "Develop practical cybersecurity projects"
        ]
    },


    "Database Administrator": {

        "description":
            "Manages databases, including their availability, organization, security and performance.",

        "skills": [
            "Data Structures",
            "Programming",
            "Operating Systems",
            "Analytical Reasoning"
        ],

        "roadmap": [
            "Learn database fundamentals",
            "Learn SQL",
            "Understand database design",
            "Learn database administration concepts",
            "Practice with database projects"
        ]
    },


    "Graphics Designer": {

        "description":
            "Creates visual content and designs for digital and communication purposes.",

        "skills": [
            "Communication",
            "Digital Electronics",
            "Analytical Reasoning"
        ],

        "roadmap": [
            "Learn design fundamentals",
            "Practice graphic design tools",
            "Study typography and visual composition",
            "Build a design portfolio",
            "Work on practical design projects"
        ]
    },


    "Hardware Engineer": {

        "description":
            "Works with the design, development, testing and maintenance of computer hardware systems.",

        "skills": [
            "Hardware",
            "Digital Electronics",
            "Mathematics",
            "Operating Systems"
        ],

        "roadmap": [
            "Learn digital electronics",
            "Understand computer architecture",
            "Study hardware components",
            "Practice circuit and hardware projects",
            "Gain practical hardware experience"
        ]
    },


    "Helpdesk Engineer": {

        "description":
            "Provides technical assistance to users and helps troubleshoot computer, software and network-related problems.",

        "skills": [
            "Communication",
            "Operating Systems",
            "Networking",
            "Hardware"
        ],

        "roadmap": [
            "Learn computer fundamentals",
            "Understand operating systems",
            "Learn networking basics",
            "Develop troubleshooting skills",
            "Practice technical support scenarios"
        ]
    },


    "Information Security Specialist": {

        "description":
            "Helps protect information systems and data by identifying and addressing information security risks.",

        "skills": [
            "Networking",
            "Operating Systems",
            "Analytical Reasoning",
            "Programming"
        ],

        "roadmap": [
            "Learn networking fundamentals",
            "Understand operating system security",
            "Study information security concepts",
            "Practice security labs",
            "Develop practical security skills"
        ]
    },


    "Networking Engineer": {

        "description":
            "Designs, configures and maintains computer networks and network infrastructure.",

        "skills": [
            "Networking",
            "Operating Systems",
            "Hardware",
            "Communication"
        ],

        "roadmap": [
            "Learn networking fundamentals",
            "Understand network protocols",
            "Practice network configuration",
            "Learn network troubleshooting",
            "Build practical networking experience"
        ]
    },


    "Project Manager": {

        "description":
            "Plans, coordinates and manages projects while working with teams to achieve project goals.",

        "skills": [
            "Communication",
            "Analytical Reasoning",
            "Programming",
            "Mathematics"
        ],

        "roadmap": [
            "Develop communication skills",
            "Learn project management fundamentals",
            "Understand project planning",
            "Practice team and task management",
            "Gain experience working on projects"
        ]
    },


    "Software Developer": {

        "description":
            "Designs, develops, tests and maintains software applications and systems.",

        "skills": [
            "Programming",
            "Data Structures",
            "Operating Systems",
            "Analytical Reasoning"
        ],

        "roadmap": [
            "Learn programming fundamentals",
            "Strengthen data structures and algorithms",
            "Learn software development concepts",
            "Build practical projects",
            "Gain internship or development experience"
        ]
    },


    "Software Tester": {

        "description":
            "Tests software applications to identify defects and verify that applications work as expected.",

        "skills": [
            "Analytical Reasoning",
            "Programming",
            "Communication",
            "Data Structures"
        ],

        "roadmap": [
            "Learn software development basics",
            "Understand software testing concepts",
            "Learn test case design",
            "Practice manual and automated testing",
            "Build practical testing experience"
        ]
    },


    "Technical Writer": {

        "description":
            "Creates technical documentation that explains software, systems, processes and technical information clearly.",

        "skills": [
            "Communication",
            "Programming",
            "Operating Systems",
            "Analytical Reasoning"
        ],

        "roadmap": [
            "Improve technical writing skills",
            "Understand basic programming concepts",
            "Learn documentation tools",
            "Practice writing technical documentation",
            "Build a technical writing portfolio"
        ]
    }

}


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():

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

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


# Initialize database
init_db()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------
    # Read and validate input
    # -----------------------------------------

    try:

        branch = request.form["branch"]

        skills = {

            field: int(
                request.form[field]
            )

            for field in SKILL_FIELDS
        }

    except (
        KeyError,
        ValueError
    ):

        return (
            "Please fill in all fields correctly.",
            400
        )


    # -----------------------------------------
    # Validate rating range
    # -----------------------------------------

    if any(
        value < 0 or value > 4
        for value in skills.values()
    ):

        return (
            "Ratings must be between 0 and 4.",
            400
        )


    # -----------------------------------------
    # Create student dataframe
    # -----------------------------------------

    student = pd.DataFrame([

        {
            "branch": branch,
            **skills
        }

    ])


    # -----------------------------------------
    # Predict career
    # -----------------------------------------

    prediction = model.predict(
        student
    )[0]


    # -----------------------------------------
    # Prediction probabilities
    # -----------------------------------------

    probabilities = model.predict_proba(
        student
    )[0]


    # -----------------------------------------
    # Sort predictions
    # -----------------------------------------

    results = sorted(

        zip(
            model.classes_,
            probabilities
        ),

        key=lambda x: x[1],

        reverse=True
    )


    # -----------------------------------------
    # Top 5 predictions
    # -----------------------------------------

    top_results = results[:5]


    # =====================================================
    # EXPLAINABILITY
    # =====================================================

    rf_model = model.named_steps["model"]

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    transformed_features = (
        preprocessor.get_feature_names_out()
    )

    importances = (
        rf_model.feature_importances_
    )

    feature_importance_data = []


    for feature_name, importance in zip(
        transformed_features,
        importances
    ):

        # Ignore branch encoded features
        if feature_name.startswith(
            "branch_"
        ):

            continue


        # Remove pipeline prefix
        clean_name = feature_name.replace(
            "remainder__",
            ""
        )


        # Convert to readable name
        display_name = (
            SKILL_DISPLAY_NAMES.get(
                clean_name,
                clean_name.replace(
                    "_",
                    " "
                ).title()
            )
        )


        feature_importance_data.append(

            {
                "skill": display_name,

                "importance":
                    float(importance)
            }
        )


    # -----------------------------------------
    # Sort feature importance
    # -----------------------------------------

    feature_importance_data.sort(

        key=lambda x:
            x["importance"],

        reverse=True
    )


    # -----------------------------------------
    # Top 5 features
    # -----------------------------------------

    top_features = (
        feature_importance_data[:5]
    )


    # -----------------------------------------
    # Convert to percentage
    # -----------------------------------------

    for feature in top_features:

        feature["percentage"] = round(

            feature["importance"] * 100,

            2
        )


    # =====================================================
    # CAREER DETAILS
    # =====================================================

    career_details = CAREER_DETAILS.get(

        prediction,

        {

            "description":
                "Career information is not available for this prediction.",

            "skills": [],

            "roadmap": []
        }
    )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        prediction=prediction,

        results=top_results,

        top_features=top_features,

        career_details=career_details
    )


# =========================================================
# FEEDBACK
# =========================================================

@app.route(
    "/feedback",
    methods=["GET", "POST"]
)
def feedback():

    # -----------------------------------------
    # Show feedback page
    # -----------------------------------------

    if request.method == "GET":

        return render_template(
            "feedback.html"
        )


    # -----------------------------------------
    # Receive feedback
    # -----------------------------------------

    name = request.form.get(
        "name",
        ""
    ).strip()


    accuracy = request.form.get(
        "accuracy",
        ""
    ).strip()


    ease_of_use = request.form.get(
        "ease_of_use",
        ""
    ).strip()


    consider_career = request.form.get(
        "consider_career",
        ""
    ).strip()


    message = request.form.get(
        "message",
        ""
    ).strip()


    # -----------------------------------------
    # Validate feedback
    # -----------------------------------------

    if (

        not accuracy

        or not ease_of_use

        or not consider_career

        or not message

    ):

        return (
            "Please complete all required feedback fields.",
            400
        )


    # -----------------------------------------
    # Thank you page
    # -----------------------------------------

    return """

    <!DOCTYPE html>

    <html lang="en">

    <head>

        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <title>
            Thank You | CareerGuide
        </title>


        <style>

            body {

                font-family:
                    Arial,
                    sans-serif;

                text-align: center;

                padding: 80px 20px;

                background:
                    #f8fafc;
            }


            .thank-you {

                max-width: 600px;

                margin: auto;

                background: white;

                padding: 50px 30px;

                border-radius: 16px;

                box-shadow:
                    0 10px 30px
                    rgba(
                        0,
                        0,
                        0,
                        0.08
                    );
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


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )