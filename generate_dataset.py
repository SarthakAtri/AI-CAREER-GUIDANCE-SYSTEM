import pandas as pd
import numpy as np

np.random.seed(42)

# --------------------------------------------------
# Career categories from the detailed classification
# in the review paper
# --------------------------------------------------

careers = [
    "AI/ML Specialist",
    "API Specialist",
    "Application Support Engineer",
    "Business Analyst",
    "Customer Service Executive",
    "Cyber Security Specialist",
    "Database Administrator",
    "Graphics Designer",
    "Hardware Engineer",
    "Helpdesk Engineer",
    "Information Security Specialist",
    "Networking Engineer",
    "Project Manager",
    "Software Developer",
    "Software Tester",
    "Technical Writer"
]

branches = ["CSE", "IT", "ECE"]

skills = [
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

# --------------------------------------------------
# Synthetic skill profiles
#
# Values represent the approximate average skill
# level (0-4) used to generate project data.
#
# These mappings are project assumptions and are
# not claimed to be taken directly from the papers.
# --------------------------------------------------

career_profiles = {

    "AI/ML Specialist":
        [4, 4, 1, 4, 3, 4, 3, 2, 1, 4],

    "API Specialist":
        [4, 3, 1, 2, 3, 4, 3, 3, 1, 2],

    "Application Support Engineer":
        [3, 3, 1, 2, 3, 2, 4, 3, 1, 1],

    "Business Analyst":
        [2, 4, 1, 3, 4, 2, 2, 2, 1, 2],

    "Customer Service Executive":
        [1, 2, 0, 1, 4, 1, 1, 1, 0, 0],

    "Cyber Security Specialist":
        [3, 4, 2, 3, 3, 3, 4, 4, 1, 2],

    "Database Administrator":
        [3, 3, 1, 2, 3, 4, 4, 3, 1, 1],

    "Graphics Designer":
        [1, 2, 1, 1, 3, 1, 1, 0, 1, 0],

    "Hardware Engineer":
        [2, 3, 4, 4, 2, 1, 3, 2, 4, 1],

    "Helpdesk Engineer":
        [2, 3, 2, 2, 4, 1, 3, 3, 1, 1],

    "Information Security Specialist":
        [3, 4, 2, 3, 3, 3, 4, 4, 2, 2],

    "Networking Engineer":
        [3, 3, 3, 3, 2, 2, 3, 4, 2, 1],

    "Project Manager":
        [2, 4, 1, 3, 4, 2, 2, 2, 1, 1],

    "Software Developer":
        [4, 4, 1, 3, 3, 4, 3, 2, 1, 2],

    "Software Tester":
        [3, 4, 1, 2, 3, 3, 3, 2, 1, 1],

    "Technical Writer":
        [2, 3, 1, 2, 4, 2, 2, 1, 1, 1]
}


# --------------------------------------------------
# Generate dataset
# --------------------------------------------------

records_per_career = 125

data = []

for career in careers:

    profile = career_profiles[career]

    for _ in range(records_per_career):

        # Select branch
        branch = np.random.choice(branches)

        # Add some random variation around the
        # career's skill profile.
        ratings = np.random.normal(
            loc=profile,
            scale=0.75
        )

        # Keep ratings between 0 and 4
        ratings = np.clip(
            np.round(ratings),
            0,
            4
        ).astype(int)

        row = {
            "branch": branch,
            "programming": ratings[0],
            "analytical_reasoning": ratings[1],
            "hardware": ratings[2],
            "mathematics": ratings[3],
            "communication": ratings[4],
            "data_structures": ratings[5],
            "operating_systems": ratings[6],
            "networking": ratings[7],
            "digital_electronics": ratings[8],
            "machine_learning": ratings[9],
            "career": career
        }

        data.append(row)


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

df = pd.DataFrame(data)

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset
df.to_csv("career_dataset.csv", index=False)


# --------------------------------------------------
# Display information
# --------------------------------------------------

print("Dataset created successfully!")
print(f"Total records: {len(df)}")
print(f"Total careers: {df['career'].nunique()}")

print("\nCareer distribution:")
print(df["career"].value_counts())

print("\nFirst 5 records:")
print(df.head())