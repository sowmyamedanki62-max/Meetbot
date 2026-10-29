import hashlib
import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import extra_streamlit_components as stx
import streamlit as st

from services.transcription import transcribe_audio
from services.meeting_agent import generate_meeting_minutes
from utils.export import create_meeting_document


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MeetBot | AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).resolve().parent / "meetbot.db"
COOKIE_NAME = "meetbot_auth_token"
SESSION_DAYS = 30


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                transcript TEXT NOT NULL,
                minutes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Persistent login sessions. Only token hashes are stored.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        300_000,
    )

    return password_hash.hex(), salt.hex()


def create_user(full_name, email, password):
    password_hash, salt = hash_password(password)

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                (full_name, email, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    full_name,
                    email,
                    password_hash,
                    salt,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

        return True, "Account created successfully! Please sign in."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate_user(email, password):
    with get_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if user is None:
        return None

    stored_hash, _ = hash_password(
        password,
        bytes.fromhex(user["password_salt"]),
    )

    if not hmac.compare_digest(stored_hash, user["password_hash"]):
        return None

    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
    }


# --------------------------------------------------
# PERSISTENT LOGIN
# --------------------------------------------------

def create_login_session(user_id):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    now = datetime.now()
    expires_at = now + timedelta(days=SESSION_DAYS)

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?",
            (now.strftime("%Y-%m-%d %H:%M:%S"),),
        )

        connection.execute(
            """
            INSERT INTO auth_sessions
            (user_id, token_hash, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                token_hash,
                expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                now.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

    return token, expires_at


def get_user_from_login_token(token):
    if not token or not isinstance(token, str):
        return None

    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT users.id, users.full_name, users.email
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
              AND auth_sessions.expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()

    if user is None:
        return None

    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
    }


def revoke_login_session(token):
    if not token or not isinstance(token, str):
        return

    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM auth_sessions WHERE token_hash = ?",
            (token_hash,),
        )


# --------------------------------------------------
# MEETING HISTORY DATABASE FUNCTIONS
# --------------------------------------------------

def save_meeting(user_id, title, transcript, minutes):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO meetings
            (user_id, title, transcript, minutes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                title,
                transcript,
                minutes,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        return cursor.lastrowid


def update_meeting(meeting_id, user_id, title, transcript, minutes):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE meetings
            SET title = ?, transcript = ?, minutes = ?
            WHERE id = ? AND user_id = ?
            """,
            (title, transcript, minutes, meeting_id, user_id),
        )


def get_user_meetings(user_id):
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, title, created_at
            FROM meetings
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()


def get_meeting_for_user(meeting_id, user_id):
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT *
            FROM meetings
            WHERE id = ? AND user_id = ?
            """,
            (meeting_id, user_id),
        ).fetchone()


initialize_database()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "user": None,
    "page": "Create Meeting Minutes",
    "transcript": "",
    "minutes": "",
    "transcript_editor": "",
    "meeting_title": "Team Meeting",
    "selected_history_id": None,
    "saved_meeting_id": None,
    "signup_success_message": "",
    "auth_token": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# RESTORE LOGIN FROM BROWSER COOKIE
# --------------------------------------------------

cookie_manager = stx.CookieManager(key="meetbot_cookie_manager")
cookies = cookie_manager.get_all()

# Wait for the cookie component to load before deciding
# whether the user needs to sign in.
if cookies is None:
    st.info("Restoring your session...")
    st.stop()

if st.session_state.user is None:
    browser_token = cookies.get(COOKIE_NAME)

    if browser_token:
        restored_user = get_user_from_login_token(browser_token)

        if restored_user:
            st.session_state.user = restored_user
            st.session_state.auth_token = browser_token
        else:
            # Invalid or expired tokens cannot authenticate the user.
            try:
                cookie_manager.delete(COOKIE_NAME)
            except Exception:
                pass


# --------------------------------------------------
# STYLING — REFINED PURPLE AND WHITE THEME
# --------------------------------------------------

st.markdown("""
<style>
    :root {
        --mb-purple: #6847df;
        --mb-purple-dark: #5032bd;
        --mb-purple-light: #f0ecff;
        --mb-ink: #292044;
        --mb-muted: #716985;
        --mb-border: #e8e2f7;
        --mb-white: #ffffff;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #f7f6fc;
        color: var(--mb-ink);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp h5, .stApp h6 {
        color: var(--mb-ink) !important;
        letter-spacing: -0.02em;
    }

    .stApp p, .stApp label,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMarkdownContainer"] {
        color: #38304f;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--mb-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] * {
        color: #302653;
    }

    .brand {
        font-size: 30px;
        font-weight: 850;
        letter-spacing: -1px;
        color: var(--mb-purple) !important;
        margin-bottom: 5px;
    }

    .hero {
        background: linear-gradient(125deg, #5838ce 0%, #7655e8 52%, #a18af7 100%);
        padding: 36px 38px;
        border-radius: 22px;
        color: #ffffff;
        margin-bottom: 28px;
        box-shadow: 0 12px 28px rgba(91, 61, 190, 0.16);
        position: relative;
        overflow: hidden;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(1.8rem, 3vw, 2.45rem);
        font-weight: 800;
        margin: 0 0 10px 0;
        line-height: 1.2;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.94) !important;
        font-size: 1.03rem;
        line-height: 1.65;
        margin: 0;
        max-width: 850px;
    }

    .feature-card {
        background: var(--mb-white);
        border: 1px solid var(--mb-border);
        border-radius: 17px;
        padding: 23px;
        min-height: 150px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(53, 37, 104, 0.045);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        border-color: #c9bcf7;
        box-shadow: 0 10px 24px rgba(83, 52, 196, 0.10);
    }

    .feature-card h3 {
        color: var(--mb-purple) !important;
        font-size: 1.12rem;
        font-weight: 750;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .feature-card p {
        color: #625a76 !important;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(115deg, #6847df, #7958e8);
        color: #ffffff !important;
        border: 1px solid transparent;
        border-radius: 11px;
        min-height: 44px;
        padding: 0.55rem 1rem;
        font-weight: 650;
        box-shadow: 0 4px 10px rgba(104, 71, 223, 0.12);
        transition: background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button p,
    .stDownloadButton > button p,
    .stFormSubmitButton > button p {
        color: #ffffff !important;
        font-weight: 650;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(115deg, #5032bd, #6847df);
        color: #ffffff !important;
        border-color: transparent;
        transform: translateY(-1px);
        box-shadow: 0 7px 15px rgba(83, 52, 196, 0.20);
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus,
    .stFormSubmitButton > button:focus {
        outline: 3px solid rgba(104, 71, 223, 0.22);
        outline-offset: 2px;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1.5px dashed #a996ed;
        border-radius: 15px;
        padding: 16px;
        box-shadow: 0 3px 12px rgba(53, 37, 104, 0.035);
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: #302653 !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--mb-border);
        border-radius: 15px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(53, 37, 104, 0.04);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: var(--mb-muted) !important;
        font-size: 0.92rem !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: var(--mb-ink) !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: #24213b !important;
    }

    [data-testid="stAlert"][kind="success"] {
        background: #e8f9ee !important;
        border-color: #a7e8bd !important;
    }

    [data-testid="stAlert"][kind="info"] {
        background: #f0ecff !important;
        border-color: #d1c5ff !important;
    }

    [data-testid="stAlert"][kind="warning"] {
        background: #fff7df !important;
        border-color: #f3d98b !important;
    }

    [data-testid="stAlert"][kind="error"] {
        background: #fff0f0 !important;
        border-color: #f3b4b4 !important;
    }

    .stTextArea textarea,
    .stTextInput input {
        background: #ffffff !important;
        color: #24213b !important;
        border: 1px solid #d8d1ed !important;
        border-radius: 11px;
        caret-color: var(--mb-purple);
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--mb-purple) !important;
        box-shadow: 0 0 0 2px rgba(104, 71, 223, 0.13) !important;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #827895 !important;
    }

    [data-testid="stSelectbox"] {
        color: #302653;
    }

    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p {
        color: #302653 !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--mb-border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        font-weight: 650;
    }

    .footer {
        text-align: center;
        color: #827b95 !important;
        padding: 30px 0 12px;
        font-size: 0.84rem;
    }

    hr {
        border-color: #e8e2f7;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2rem;
        }
        .hero {
            padding: 25px 22px;
            border-radius: 17px;
        }
        .feature-card {
            padding: 18px;
            min-height: auto;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# REUSABLE COMPONENTS
# --------------------------------------------------

def show_footer():
    st.markdown(
        '<div class="footer">MeetBot | Powered by Whisper and Ollama</div>',
        unsafe_allow_html=True,
    )


def show_hero(title, description):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# LOGIN AND SIGNUP
# --------------------------------------------------

if st.session_state.user is None:

    st.markdown(
        '<div class="brand">🎙️ MeetBot</div>',
        unsafe_allow_html=True,
    )

    st.caption("Your AI-powered meeting minutes assistant")
    st.markdown("---")

    left, center, right = st.columns([1, 1.4, 1])

    with center:
        st.markdown("## Welcome to MeetBot")

        st.write(
            "Sign in to create meeting minutes and access "
            "your saved meeting history."
        )

        auth_choice = st.radio(
            "Choose an option",
            ["Login", "Sign Up"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_choice",
        )

        if st.session_state.get("signup_success_message"):
            st.success(st.session_state["signup_success_message"])
            st.info(
                "Your account is ready! Select Login above to sign in."
            )

            if st.button("Go to Login", key="go_to_login"):
                st.session_state["signup_success_message"] = ""
                st.rerun()

        if auth_choice == "Sign Up":
            st.subheader("Create your account")

            with st.form("signup_form"):
                full_name = st.text_input(
                    "Full name",
                    placeholder="Enter your full name",
                )

                email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="At least 8 characters",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                )

                signup_submit = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                )

            if signup_submit:
                full_name = full_name.strip()
                email = email.strip().lower()

                if not full_name or not email or not password:
                    st.error("Please fill in all required fields.")

                elif not re.fullmatch(
                    r"[^@\s]+@[^@\s]+\.[^@\s]+",
                    email,
                ):
                    st.error("Please enter a valid email address.")

                elif len(password) < 8:
                    st.error(
                        "Password must contain at least 8 characters."
                    )

                elif password != confirm_password:
                    st.error("The passwords do not match.")

                else:
                    success, message = create_user(
                        full_name,
                        email,
                        password,
                    )

                    if success:
                        st.session_state["signup_success_message"] = message
                        st.rerun()
                    else:
                        st.error(message)

        else:
            st.subheader("Sign in to your account")

            with st.form("login_form"):
                email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                )

                login_submit = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                )

            if login_submit:
                email = email.strip().lower()
                user = authenticate_user(email, password)

                if user:
                    token, expires_at = create_login_session(user["id"])

                    st.session_state.user = user
                    st.session_state.auth_token = token
                    st.session_state.page = "Create Meeting Minutes"
                    st.session_state["signup_success_message"] = ""

                    cookie_manager.set(
                        COOKIE_NAME,
                        token,
                        expires_at=expires_at,
                        key="set_auth_cookie",
                    )

                    st.rerun()

                else:
                    st.error("Invalid email or password.")

    show_footer()
    st.stop()


# --------------------------------------------------
# LOGGED-IN SIDEBAR
# --------------------------------------------------

current_user = st.session_state.user

with st.sidebar:
    st.markdown(
        '<div class="brand">🎙️ MeetBot</div>',
        unsafe_allow_html=True,
    )

    # Show the registered name, not the email as the greeting.
    st.subheader(f"Welcome, {current_user['full_name']}!")
    st.caption(current_user["email"])
    st.markdown("---")

    selected_page = st.radio(
        "Menu",
        ["Create Meeting Minutes", "Meeting History"],
        index=(
            0
            if st.session_state.page == "Create Meeting Minutes"
            else 1
        ),
        label_visibility="collapsed",
        key="main_page_radio",
    )

    st.session_state.page = selected_page

    st.markdown("---")
    st.caption("Speech-to-text • AI summaries • Export")

    if st.button("Log Out", use_container_width=True, key="logout_button"):
        token_to_revoke = (
            st.session_state.get("auth_token")
            or cookies.get(COOKIE_NAME)
        )

        # Revoke the token before clearing the browser cookie.
        revoke_login_session(token_to_revoke)

        st.session_state.user = None
        st.session_state.auth_token = None
        st.session_state.transcript = ""
        st.session_state.minutes = ""
        st.session_state.transcript_editor = ""
        st.session_state.meeting_title = "Team Meeting"
        st.session_state.saved_meeting_id = None
        st.session_state.selected_history_id = None
        st.session_state.page = "Create Meeting Minutes"
        st.session_state["signup_success_message"] = ""

        try:
            cookie_manager.delete(COOKIE_NAME)
        except Exception:
            pass

        st.rerun()


# --------------------------------------------------
# MEETING HISTORY
# --------------------------------------------------

if st.session_state.page == "Meeting History":

    show_hero(
        "Meeting History",
        "Your previously generated meeting minutes, all in one place.",
    )

    # Each account sees only its own saved meetings.
    meetings = get_user_meetings(current_user["id"])

    if not meetings:
        st.info(
            "You haven't saved any meetings yet. "
            "Create meeting minutes to see them here."
        )

    else:
        meeting_options = {
            f"{meeting['title']} — {meeting['created_at']}":
            meeting["id"]
            for meeting in meetings
        }

        labels = list(meeting_options.keys())

        selected_label = st.selectbox(
            "Select a saved meeting",
            labels,
        )

        selected_id = meeting_options[selected_label]

        meeting = get_meeting_for_user(
            selected_id,
            current_user["id"],
        )

        if meeting:
            st.subheader(meeting["title"])
            st.caption(f"Saved on {meeting['created_at']}")

            st.markdown("### Generated Meeting Minutes")
            st.markdown(meeting["minutes"])

            with st.expander("View meeting transcript"):
                st.text_area(
                    "Transcript",
                    value=meeting["transcript"],
                    height=220,
                    disabled=True,
                    key=f"history_transcript_{meeting['id']}",
                )

            st.download_button(
                "Download Meeting Minutes",
                data=(
                    f"Meeting: {meeting['title']}\n"
                    f"Date: {meeting['created_at']}\n\n"
                    f"MEETING MINUTES\n\n{meeting['minutes']}\n\n"
                    f"TRANSCRIPT\n\n{meeting['transcript']}"
                ),
                file_name=f"meeting_{meeting['id']}_minutes.txt",
                mime="text/plain",
                key=f"history_txt_download_{meeting['id']}",
            )

            history_word_document = create_meeting_document(
                title=meeting["title"],
                meeting_date=meeting["created_at"],
                user_name=current_user["full_name"],
                minutes=meeting["minutes"],
                transcript=meeting["transcript"],
            )

            st.download_button(
                label="Download as Word Document (.docx)",
                data=history_word_document,
                file_name=f"meeting_{meeting['id']}_minutes.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                key=f"history_word_download_{meeting['id']}",
            )

            if st.button("Open this meeting in the editor"):
                st.session_state.meeting_title = meeting["title"]
                st.session_state.transcript = meeting["transcript"]
                st.session_state.transcript_editor = meeting["transcript"]
                st.session_state.minutes = meeting["minutes"]
                st.session_state.saved_meeting_id = meeting["id"]
                st.session_state.page = "Create Meeting Minutes"
                st.rerun()

    show_footer()
    st.stop()


# --------------------------------------------------
# CREATE MEETING MINUTES DASHBOARD
# --------------------------------------------------

show_hero(
    f"Welcome back, {current_user['full_name']}!",
    "Turn conversations into action. Upload your meeting audio, "
    "generate AI-powered minutes, and keep your team on track.",
)

# Dashboard overview: only count meetings belonging to the signed-in user.
dashboard_meetings = get_user_meetings(current_user["id"])
meeting_count = len(dashboard_meetings)
latest_meeting = dashboard_meetings[0] if dashboard_meetings else None

st.markdown("### Your Meeting Overview")
stat1, stat2, stat3 = st.columns(3)
with stat1:
    st.metric("Saved Meetings", meeting_count)
with stat2:
    st.metric("Transcript", "Ready" if st.session_state.transcript else "Not started")
with stat3:
    st.metric("Meeting Minutes", "Generated" if st.session_state.minutes else "Not generated")

if latest_meeting:
    st.markdown("#### Recent Meeting")
    st.markdown(
        f"**{latest_meeting['title']}**  \n"
        f"<span style='color:#716985;font-size:0.9rem'>"
        f"Last saved: {latest_meeting['created_at']}</span>",
        unsafe_allow_html=True,
    )

    if len(dashboard_meetings) > 1:
        with st.expander("View more recent meetings"):
            for recent_meeting in dashboard_meetings[1:4]:
                st.markdown(
                    f"- **{recent_meeting['title']}** — "
                    f"{recent_meeting['created_at']}"
                )

    if st.button("Open Meeting History", key="dashboard_history_button"):
        st.session_state.page = "Meeting History"
        st.session_state.main_page_radio = "Meeting History"
        st.rerun()
else:
    st.caption("Your saved meetings will appear here after you generate your first set of minutes.")

st.markdown("### What MeetBot can do")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🎧 Speech to Text</h3>
        <p>Convert meeting recordings into readable transcripts.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI Analysis</h3>
        <p>Generate organized meeting minutes from your transcript.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📥 Export Results</h3>
        <p>Download your meeting minutes and transcript as TXT or Word.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("## 1. Meeting Details")

st.session_state.meeting_title = st.text_input(
    "Meeting title",
    value=st.session_state.meeting_title,
    placeholder="e.g. Weekly Team Meeting",
)

st.markdown("## 2. Upload Meeting Audio")

uploaded_file = st.file_uploader(
    "Choose an audio or video recording",
    type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm"],
)

if st.button("Transcribe Audio", use_container_width=True):

    if uploaded_file is None:
        st.warning("Please upload an audio file first.")

    else:
        try:
            with st.spinner("Transcribing your recording..."):
                transcript_result = transcribe_audio(uploaded_file)

            if isinstance(transcript_result, dict):
                transcript_result = transcript_result.get("text", "")

            st.session_state.transcript = str(transcript_result).strip()
            st.session_state.transcript_editor = st.session_state.transcript
            st.session_state.minutes = ""
            st.session_state.saved_meeting_id = None

            if st.session_state.transcript:
                st.success("Transcription completed successfully!")
            else:
                st.warning("No speech was detected. Try another recording.")

        except Exception as error:
            st.error(f"Transcription failed: {error}")


st.markdown("## 3. Review Transcript")

if st.session_state.transcript:

    st.session_state.transcript_editor = st.text_area(
        "Edit your transcript if needed",
        value=st.session_state.transcript_editor,
        height=250,
        key="editable_transcript",
    )

    word_count = len(st.session_state.transcript_editor.split())
    character_count = len(st.session_state.transcript_editor)

    metric1, metric2 = st.columns(2)
    metric1.metric("Words", word_count)
    metric2.metric("Characters", character_count)

    if st.button("Generate Meeting Minutes", use_container_width=True):

        edited_transcript = st.session_state.transcript_editor.strip()

        if not edited_transcript:
            st.warning("Please enter a transcript first.")

        else:
            try:
                with st.spinner("AI is generating your meeting minutes..."):
                    generated_minutes = generate_meeting_minutes(
                        edited_transcript
                    )

                if isinstance(generated_minutes, dict):
                    generated_minutes = (
                        generated_minutes.get("minutes")
                        or generated_minutes.get("text")
                        or str(generated_minutes)
                    )

                generated_minutes = str(generated_minutes).strip()

                if not generated_minutes:
                    st.error(
                        "The AI returned empty minutes. Please try again."
                    )

                else:
                    st.session_state.transcript = edited_transcript
                    st.session_state.transcript_editor = edited_transcript
                    st.session_state.minutes = generated_minutes

                    title = (
                        st.session_state.meeting_title.strip()
                        or "Untitled Meeting"
                    )

                    if st.session_state.saved_meeting_id is None:
                        saved_id = save_meeting(
                            current_user["id"],
                            title,
                            edited_transcript,
                            generated_minutes,
                        )

                        st.session_state.saved_meeting_id = saved_id

                    else:
                        update_meeting(
                            st.session_state.saved_meeting_id,
                            current_user["id"],
                            title,
                            edited_transcript,
                            generated_minutes,
                        )

                    st.success(
                        "Meeting minutes generated and saved to your history!"
                    )

            except Exception as error:
                st.error(f"Could not generate meeting minutes: {error}")

else:
    st.info(
        "Upload your meeting recording and click Transcribe Audio "
        "to get started."
    )


# --------------------------------------------------
# DISPLAY AND DOWNLOAD MINUTES
# --------------------------------------------------

if st.session_state.minutes:

    st.markdown("---")
    st.markdown("## 4. Generated Meeting Minutes")
    st.markdown(st.session_state.minutes)

    title = (
        st.session_state.meeting_title.strip()
        or "Untitled Meeting"
    )

    meeting_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    download_content = (
        f"MEETING: {title}\n"
        f"DATE: {meeting_date}\n"
        f"USER: {current_user['full_name']}\n\n"
        f"MEETING MINUTES\n\n{st.session_state.minutes}\n\n"
        f"TRANSCRIPT\n\n{st.session_state.transcript_editor}"
    )

    st.download_button(
        "Download Meeting Minutes and Transcript",
        data=download_content,
        file_name="meetbot_meeting_minutes.txt",
        mime="text/plain",
        use_container_width=True,
    )

    word_document = create_meeting_document(
        title=title,
        meeting_date=meeting_date,
        user_name=current_user["full_name"],
        minutes=st.session_state.minutes,
        transcript=st.session_state.transcript_editor,
    )

    st.download_button(
        label="Download as Word Document (.docx)",
        data=word_document,
        file_name="meetbot_meeting_minutes.docx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        use_container_width=True,
    )


show_footer()
