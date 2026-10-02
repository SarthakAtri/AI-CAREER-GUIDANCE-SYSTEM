import sqlite3


def create_database():

    connection = sqlite3.connect("career_guidance.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            branch TEXT NOT NULL,

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
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_database()

    print("Database created successfully!")