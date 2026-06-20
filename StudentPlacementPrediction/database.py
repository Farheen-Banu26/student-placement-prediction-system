"""
database.py
-----------
SQLite database functions for the Student Placement Prediction System.
Manages users table and prediction history table in student.db.
"""

import sqlite3
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "student.db")


def get_connection():
    """Create and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def create_tables():
    """
    Create the required database tables if they do not already exist.
    Tables:
        - users: stores registered user information
        - prediction_history: stores past prediction results per user
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            email       TEXT    NOT NULL UNIQUE,
            password    TEXT    NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # Create prediction history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            cgpa            REAL,
            branch          TEXT,
            internships     INTEGER,
            projects        INTEGER,
            coding_score    REAL,
            aptitude_score  REAL,
            comm_score      REAL,
            prediction      TEXT,
            probability     REAL,
            predicted_at    TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def insert_user(username: str, email: str, hashed_password: str) -> bool:
    """
    Insert a new user into the users table.

    Args:
        username (str): Desired username.
        email (str): User's email address.
        hashed_password (str): Bcrypt-hashed password string.

    Returns:
        bool: True if insertion succeeded, False if username/email already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Duplicate username or email
        return False
    finally:
        conn.close()


def validate_login(username: str) -> dict | None:
    """
    Fetch user record by username for login validation.

    Args:
        username (str): The username to look up.

    Returns:
        dict | None: User record dict if found, else None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, password FROM users WHERE username = ?",
        (username,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def store_prediction(
    user_id: int,
    cgpa: float,
    branch: str,
    internships: int,
    projects: int,
    coding_score: float,
    aptitude_score: float,
    comm_score: float,
    prediction: str,
    probability: float,
) -> None:
    """
    Store a prediction result in the prediction_history table.

    Args:
        user_id (int): ID of the logged-in user.
        cgpa (float): Student CGPA.
        branch (str): Student branch/department.
        internships (int): Number of internships done.
        projects (int): Number of projects done.
        coding_score (float): Coding skill score.
        aptitude_score (float): Aptitude score.
        comm_score (float): Communication skill score.
        prediction (str): 'Placed' or 'Not Placed'.
        probability (float): Probability percentage.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prediction_history
            (user_id, cgpa, branch, internships, projects,
             coding_score, aptitude_score, comm_score, prediction, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            cgpa,
            branch,
            internships,
            projects,
            coding_score,
            aptitude_score,
            comm_score,
            prediction,
            probability,
        ),
    )
    conn.commit()
    conn.close()


def get_user_predictions(user_id: int) -> list[dict]:
    """
    Retrieve all past predictions for a given user.

    Args:
        user_id (int): ID of the logged-in user.

    Returns:
        list[dict]: List of prediction record dicts, newest first.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT cgpa, branch, internships, projects,
               coding_score, aptitude_score, comm_score,
               prediction, probability, predicted_at
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY predicted_at DESC
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Initialize tables on import
create_tables()
