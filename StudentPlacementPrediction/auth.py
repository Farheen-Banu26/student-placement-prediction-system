"""
auth.py
-------
Authentication system for the Student Placement Prediction Streamlit App.
Handles signup, login, logout, and session state management using bcrypt for
secure password hashing.
"""

import bcrypt
import streamlit as st
from database import insert_user, validate_login


# ---------------------------------------------------------------------------
# Password Hashing Helpers
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Args:
        plain_password (str): The raw password entered by the user.

    Returns:
        str: The bcrypt-hashed password as a UTF-8 string.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a stored bcrypt hash.

    Args:
        plain_password (str): Password entered during login.
        hashed_password (str): Bcrypt hash stored in the database.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------

def init_session():
    """Initialize Streamlit session state keys related to authentication."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None


def is_logged_in() -> bool:
    """Return True if a user is currently logged in."""
    init_session()
    return st.session_state.logged_in


def get_current_user() -> dict | None:
    """
    Return basic info about the currently logged-in user.

    Returns:
        dict | None: {'user_id': int, 'username': str} or None.
    """
    if is_logged_in():
        return {
            "user_id": st.session_state.user_id,
            "username": st.session_state.username,
        }
    return None


# ---------------------------------------------------------------------------
# Core Auth Functions
# ---------------------------------------------------------------------------

def signup_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user account.

    Args:
        username (str): Desired username (must be unique).
        email (str): User's email address (must be unique).
        password (str): Plain-text password chosen by the user.

    Returns:
        tuple[bool, str]: (success_flag, message)
    """
    if not username or not email or not password:
        return False, "⚠️ All fields are required."

    if len(username) < 3:
        return False, "⚠️ Username must be at least 3 characters."

    if len(password) < 6:
        return False, "⚠️ Password must be at least 6 characters."

    if "@" not in email:
        return False, "⚠️ Please enter a valid email address."

    hashed_pw = hash_password(password)
    success = insert_user(username, email, hashed_pw)

    if success:
        return True, "✅ Account created successfully! Please log in."
    else:
        return False, "❌ Username or email already exists. Please choose another."


def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Authenticate a user and set session state on success.

    Args:
        username (str): Username entered by the user.
        password (str): Plain-text password entered by the user.

    Returns:
        tuple[bool, str]: (success_flag, message)
    """
    if not username or not password:
        return False, "⚠️ Please enter both username and password."

    user = validate_login(username)

    if user is None:
        return False, "❌ Username not found. Please sign up first."

    if not verify_password(password, user["password"]):
        return False, "❌ Incorrect password. Please try again."

    # Set session state
    init_session()
    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]

    return True, f"✅ Welcome back, **{user['username']}**!"


def logout_user() -> None:
    """
    Log out the current user by clearing all session state keys.
    """
    for key in ["logged_in", "user_id", "username"]:
        if key in st.session_state:
            del st.session_state[key]
