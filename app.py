import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

from dotenv import load_dotenv
from supabase import create_client


# =========================================================
# APP SETUP
# =========================================================

app = Flask(__name__)

load_dotenv()


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "career_model.pkl"
)

# =========================================================
# SUPABASE
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("SUPABASE URL:", SUPABASE_URL)
print(
    "SUPABASE KEY PREFIX:",
    SUPABASE_KEY[:20] if SUPABASE_KEY else "NONE"
)

supabase = None

if SUPABASE_URL and SUPABASE_KEY:

    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

    print("SUPABASE CLIENT: CONNECTED")

else:

    print(
        "SUPABASE CLIENT: NOT CONNECTED"
    )


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    MODEL_PATH
)


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
    "machine_learning"

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
        "Machine Learning"

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

    try:

        branch = request.form.get(
            "branch",
            ""
        ).strip()

        if not branch:

            return (
                "Please select your branch.",
                400
            )


        skills = {}

        for field in SKILL_FIELDS:

            value = request.form.get(
                field,
                ""
            )

            try:

                value = int(value)

            except ValueError:

                return (
                    "Please enter valid skill ratings.",
                    400
                )

            if value < 0 or value > 4:

                return (
                    "Skill ratings must be between 0 and 4.",
                    400
                )

            skills[field] = value


        student = pd.DataFrame([
            {
                "branch": branch,
                **skills
            }
        ])


        prediction = model.predict(
            student
        )[0]


        # =================================================
        # SAVE ASSESSMENT TO SUPABASE
        # =================================================

        if supabase is not None:

            try:

                assessment_data = {

                    "branch":
                        branch,

                    "programming":
                        skills["programming"],

                    "analytical_reasoning":
                        skills["analytical_reasoning"],

                    "hardware":
                        skills["hardware"],

                    "mathematics":
                        skills["mathematics"],

                    "communication":
                        skills["communication"],

                    "data_structures":
                        skills["data_structures"],

                    "operating_systems":
                        skills["operating_systems"],

                    "networking":
                        skills["networking"],

                    "digital_electronics":
                        skills["digital_electronics"],

                    "machine_learning":
                        skills["machine_learning"],

                    "predicted_career":
                        prediction

                }


                print(
                    "SUPABASE ASSESSMENT INSERT:"
                )

                print(
                    assessment_data
                )


                response = (
                    supabase
                    .table("assessments")
                    .insert(assessment_data)
                    .execute()
                )


                print(
                    "SUPABASE ASSESSMENT SAVED"
                )

                print(
                    "SUPABASE RESPONSE:",
                    response
                )


            except Exception as e:

                print(
                    "SUPABASE ASSESSMENT ERROR:",
                    e
                )


        probabilities = model.predict_proba(
            student
        )[0]


        results = sorted(

            zip(
                model.classes_,
                probabilities
            ),

            key=lambda x: x[1],

            reverse=True
        )


        top_results = results[:5]


        # =================================================
        # MODEL EXPLAINABILITY
        # =================================================

        top_features = []


        try:

            rf_model = model.named_steps[
                "model"
            ]

            preprocessor = model.named_steps[
                "preprocessor"
            ]


            transformed_features = (
                preprocessor
                .get_feature_names_out()
            )


            importances = (
                rf_model
                .feature_importances_
            )


            feature_importance_data = []


            for feature_name, importance in zip(
                transformed_features,
                importances
            ):

                if feature_name.startswith(
                    "branch_"
                ):

                    continue


                clean_name = feature_name.replace(
                    "remainder__",
                    ""
                )


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
                        "skill":
                            display_name,

                        "importance":
                            float(importance)
                    }
                )


            feature_importance_data.sort(
                key=lambda x:
                    x["importance"],

                reverse=True
            )


            top_features = (
                feature_importance_data[:5]
            )


            for feature in top_features:

                feature["percentage"] = round(
                    feature["importance"] * 100,
                    2
                )


        except Exception as e:

            print(
                "EXPLAINABILITY ERROR:",
                e
            )


        # =================================================
        # SKILL GAP ANALYSIS
        # =================================================

        skill_gap = []


        career_details = CAREER_DETAILS.get(

            prediction,

            {
                "description":
                    "Career information is not available for this prediction.",

                "skills": [],

                "roadmap": []
            }
        )


        required_skills = (
            career_details.get(
                "skills",
                []
            )
        )


        skill_mapping = {

            "Programming":
                "programming",

            "Analytical Reasoning":
                "analytical_reasoning",

            "Hardware":
                "hardware",

            "Mathematics":
                "mathematics",

            "Communication":
                "communication",

            "Data Structures":
                "data_structures",

            "Operating Systems":
                "operating_systems",

            "Networking":
                "networking",

            "Digital Electronics":
                "digital_electronics",

            "Machine Learning":
                "machine_learning"

        }


        for skill_name in required_skills:

            field = skill_mapping.get(
                skill_name
            )


            current_value = 0


            if field:

                current_value = skills.get(
                    field,
                    0
                )


            if current_value >= 3:

                status = "Strong"

            elif current_value >= 2:

                status = "Developing"

            else:

                status = "Needs Improvement"


            skill_gap.append(

                {
                    "skill":
                        skill_name,

                    "level":
                        current_value,

                    "status":
                        status
                }

            )


        return render_template(

            "result.html",

            prediction=prediction,

            results=top_results,

            top_features=top_features,

            career_details=career_details,

            skill_gap=skill_gap

        )


    except Exception as e:

        print(
            "PREDICTION ERROR:",
            e
        )

        return (

            "Something went wrong while generating "
            "your career recommendation. "
            "Please try again.",

            500

        )


# =========================================================
# FEEDBACK
# =========================================================

@app.route(
    "/feedback",
    methods=["GET", "POST"]
)
def feedback():

    if request.method == "GET":

        return render_template(
            "feedback.html"
        )


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


    if not accuracy:

        return (
            "Please provide an accuracy rating.",
            400
        )


    if not ease_of_use:

        return (
            "Please provide an ease-of-use rating.",
            400
        )


    if not consider_career:

        return (
            "Please select an option.",
            400
        )


    if not message:

        return (
            "Please enter your feedback.",
            400
        )


    try:

        accuracy_value = int(
            accuracy
        )

        ease_value = int(
            ease_of_use
        )

    except ValueError:

        return (
            "Invalid rating values.",
            400
        )


    if accuracy_value < 1 or accuracy_value > 5:

        return (
            "Accuracy rating must be between 1 and 5.",
            400
        )


    if ease_value < 1 or ease_value > 5:

        return (
            "Ease-of-use rating must be between 1 and 5.",
            400
        )


    if supabase is None:

        print(
            "SUPABASE ERROR: "
            "SUPABASE_URL or SUPABASE_KEY is missing."
        )

        return (
            "Feedback storage is not configured "
            "right now. Please try again later.",
            500
        )


    try:

        print(
            "SUPABASE FEEDBACK INSERT"
        )


        supabase.rpc(

            "submit_feedback",

            {

                "p_name":
                    name if name else None,

                "p_accuracy":
                    accuracy_value,

                "p_ease_of_use":
                    ease_value,

                "p_consider_career":
                    consider_career,

                "p_message":
                    message

            }

        ).execute()


        print(
            "SUPABASE FEEDBACK SAVED"
        )


    except Exception as e:

        print(
            "SUPABASE FEEDBACK ERROR"
        )

        print(
            "ERROR TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            e
        )


        return (

            "We could not save your feedback right now. "
            "Please try again later.",

            500

        )


    return render_template(
        "feedback.html",
        success=True
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return {

        "status":
            "ok",

        "application":
            "AI Career Guidance System"

    }


# =========================================================
# RUN APP
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