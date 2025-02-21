import streamlit as st
import time
import hashlib
import json
import os
import numpy as np
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from strava_api import fetch_activities

# Load environment variables correctly from Streamlit Cloud Secrets
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]
REFRESH_TOKEN = st.secrets["REFRESH_TOKEN"]
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]

# Constants
USER_DB = "user_data.json"
PATIENT_DB = "patient_data.json"
MODEL_PATH = "sepsis_best_model.pkl"

def send_telegram_message(message):
    """Send a message to the patient via Telegram."""
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.error("Telegram bot token or chat ID is missing.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        st.success("Message sent to patient on Telegram ✅")
    else:
        st.error(f"Failed to send message ❌: {response.text}")


# Constants
USER_DB = "user_data.json"  # JSON file for storing user credentials
PATIENT_DB = "patient_data.json"  # JSON file for storing patient registration
MODEL_PATH = "sepsis_best_model.pkl"  # Trained model

# Ensure the user and patient database files exist and are valid
for db_file in [USER_DB, PATIENT_DB]:
    if not os.path.exists(db_file):
        with open(db_file, "w") as f:
            json.dump({}, f)
    else:
        try:
            with open(db_file, "r") as f:
                json.load(f)  # Validate the JSON structure
        except json.JSONDecodeError:
            with open(db_file, "w") as f:
                json.dump({}, f)  # Reinitialize if corrupted


# Helper Functions
def hash_password(password):
    """Hash a password."""
    return hashlib.sha256(password.encode()).hexdigest()


def save_user_data(username, hashed_password):
    """Save user credentials."""
    with open(USER_DB, "r") as f:
        users = json.load(f)
    users[username] = hashed_password
    with open(USER_DB, "w") as f:
        json.dump(users, f)


def authenticate_user(username, password):
    """Authenticate user credentials."""
    with open(USER_DB, "r") as f:
        users = json.load(f)
    hashed_password = hash_password(password)
    return users.get(username) == hashed_password


def save_patient_data(patient_id, patient_name, chat_id):
    """Save registered patient details including Telegram Chat ID."""
    with open(PATIENT_DB, "r") as f:
        patients = json.load(f)
    
    patients[patient_id] = {"name": patient_name, "chat_id": chat_id}
    
    with open(PATIENT_DB, "w") as f:
        json.dump(patients, f)

def get_patient_chat_id(patient_id):
    """Retrieve the Telegram Chat ID for a patient."""
    with open(PATIENT_DB, "r") as f:
        patients = json.load(f)
    
    return patients.get(patient_id, {}).get("chat_id")



def is_patient_registered(patient_id):
    """Check if a patient is registered."""
    with open(PATIENT_DB, "r") as f:
        patients = json.load(f)
    return patient_id in patients


def refresh_strava_token():
    """Refresh the Strava access token and handle errors."""
    url = "https://www.strava.com/api/v3/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }

    response = requests.post(url, data=payload)
    data = response.json()  # Convert response to JSON

    if response.status_code == 200:
        new_access_token = data.get("access_token")
        new_refresh_token = data.get("refresh_token")

        if new_access_token:
            # Update the .env file with new tokens
            with open(".env", "w") as env_file:
                env_file.write(f"CLIENT_ID={CLIENT_ID}\n")
                env_file.write(f"CLIENT_SECRET={CLIENT_SECRET}\n")
                env_file.write(f"ACCESS_TOKEN={new_access_token}\n")
                env_file.write(f"REFRESH_TOKEN={new_refresh_token}\n")

            st.success("Strava token refreshed successfully!")
            return new_access_token, new_refresh_token
        else:
            st.error("Failed to retrieve new access token.")
            return None, None
    else:
        st.error(f"Error refreshing token: {data}")
        return None, None




def get_strava_activities():
    """Fetch the latest heart rate data from Strava."""
    access_token = refresh_strava_token()  # Ensure valid token
    if not access_token:
        return None

    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        activities = response.json()
        if activities:
            return activities  # Return all activities (latest first)
        else:
            st.warning("⚠️ No recent activity found in Strava.")
            return None
    else:
        st.error("⚠️ Failed to fetch activities from Strava.")
        return None



# App Pages
def register_page():
    """User Registration Page."""
    st.markdown("<h1 style='text-align: center; color: white; background-color: orange;'>NUB Sepsis Software Registration</h1>", unsafe_allow_html=True)

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif username in st.session_state["users"]:
            st.error("Username already exists. Please choose another.")
        else:
            hashed_password = hash_password(password)
            save_user_data(username, hashed_password)
            st.session_state["users"][username] = hashed_password
            st.success("Registration successful! You can now log in.")


def login_page():
    """User Login Page."""
    st.markdown("<h1 style='text-align: center; color: white; background-color: orange;'>NUB Sepsis Software Login</h1>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
                time.sleep(1)  # Simulate loading
            else:
                st.error("Invalid username or password.")
    with col2:
        if st.button("Create Account"):
            register_page()


def main_interface():
    """Main Interface After Login."""
    st.sidebar.write(f"Welcome, {st.session_state['username']}!")
    nav_option = st.sidebar.selectbox(
        "Navigation",
        [
            "Patient Registration",
            "Manual Prediction",
            "Real-Time Patient Monitoring",
            "Patient Medication Reminder",
            "Doctor-Patient Communication",
            "Logout",
        ],
    )

    st.markdown("<h1 style='text-align: center; color: white; background-color: orange;'>NUB Sepsis Software</h1>", unsafe_allow_html=True)

    if nav_option == "Patient Registration":
        patient_registration_interface()
    elif nav_option == "Manual Prediction":
        manual_prediction_interface()
    elif nav_option == "Real-Time Patient Monitoring":
        real_time_monitoring_interface()
    elif nav_option == "Patient Medication Reminder":
        Medication_reminders_interface()
    elif nav_option == "Doctor-Patient Communication":
        doctor_patient_communication_interface()
    elif nav_option == "Logout":
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.success("You have been logged out.")


def patient_registration_interface():
    """Patient Registration Section."""
    st.subheader("Patient Registration")
    patient_id = st.text_input("Enter Patient ID")
    patient_name = st.text_input("Enter Patient Name")

    if st.button("Register Patient"):
        if patient_id and patient_name:
            save_patient_data(patient_id, patient_name)
            st.success(f"Patient {patient_name} with ID {patient_id} has been registered.")
        else:
            st.error("Please provide both Patient ID and Patient Name.")


def manual_prediction_interface():
    """Manual Prediction Section."""
    st.subheader("Manual Prediction Section")

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    st.write("Enter Patient Data")

    # Features expected by the model (41 features)
    all_features = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2",
        "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
        "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SomeMissingFeature"  # Ensure all 41 are included
    ]

    # Collect input for each feature
    data = {}
    for feature in all_features:
        data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Sepsis Outcome"):
        input_data = np.array([data[feature] for feature in all_features]).reshape(1, -1)
        try:
            prediction = model.predict(input_data)
            outcome = "Sepsis Detected" if prediction == 1 else "No Sepsis Detected"
            st.write(f"Prediction: **{outcome}**")
        except ValueError as e:
            st.error(f"Prediction failed: {e}")

def real_time_monitoring_interface():
    """Real-Time Patient Monitoring with Strava Data & Vitals."""
    st.subheader("📡 Real-Time Patient Monitoring")
    patient_id = st.text_input("Enter Patient ID to Monitor")

    if st.button("Start Monitoring"):
        if is_patient_registered(patient_id):
            st.write("✅ Fetching latest activities...")

            activities = fetch_activities()

            if activities and len(activities) > 0:
                df = pd.DataFrame(activities)

                # ✅ Ensure required columns exist before processing
                relevant_columns = ["name", "type", "distance", "moving_time", "average_speed", "start_date_local", "average_heartrate"]
                df = df[[col for col in relevant_columns if col in df.columns]]
                df["start_date_local"] = pd.to_datetime(df["start_date_local"])

                if df.empty:
                    st.warning("⚠ No activity data found.")
                    return

                # 📌 **Display Data Table**
                st.write("### Recent Activities")
                st.dataframe(df)

                # 📊 **Distance Over Time**
                st.write("### 📈 Distance Over Time")
                fig, ax = plt.subplots()
                ax.plot(df["start_date_local"], df["distance"], marker="o", linestyle="-", color="b")
                ax.set_xlabel("Date")
                ax.set_ylabel("Distance (meters)")
                ax.set_title("Distance Over Time")

                # ✅ Fix date formatting
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate(rotation=45)

                st.pyplot(fig)

                # 📊 **Speed Over Time**
                st.write("### ⚡ Speed Over Time")
                fig, ax = plt.subplots()
                ax.plot(df["start_date_local"], df["average_speed"], marker="o", linestyle="-", color="r")
                ax.set_xlabel("Date")
                ax.set_ylabel("Speed (m/s)")
                ax.set_title("Speed Over Time")

                # ✅ Fix date formatting
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate(rotation=45)

                st.pyplot(fig)

                # 📊 **Heart Rate Over Time (if available)**
                if "average_heartrate" in df.columns:
                    st.write("### ❤️ Heart Rate Over Time")
                    fig, ax = plt.subplots()
                    ax.plot(df["start_date_local"], df["average_heartrate"], marker="o", linestyle="-", color="g")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Heart Rate (bpm)")
                    ax.set_title("Heart Rate Over Time")

                    # ✅ Fix date formatting
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    fig.autofmt_xdate(rotation=45)

                    st.pyplot(fig)

            else:
                st.error("⚠️ No activity data retrieved from Strava.")
        else:
            st.error("⚠️ Patient not registered. Please register first.")


def Medication_reminders_interface():
    """Patient Reminder Section with Telegram Notifications."""
    st.subheader("Patient Reminders")
    patient_id = st.text_input("Enter Patient ID for Reminder")
    medication = st.text_input("Enter Medication Name")
    reminder_time = st.time_input("Set Reminder Time")

    if st.button("Schedule Reminder"):
        if is_patient_registered(patient_id):
            reminder_msg = (
                f"⏰ Medication Reminder!\n"
                f"Patient ID: {patient_id}\n"
                f"Medication: {medication}\n"
                f"Reminder Time: {reminder_time}"
            )
            send_telegram_message(reminder_msg)
            st.success(f"Reminder set for Patient ID: {patient_id} and sent via Telegram.")
        else:
            st.error("Patient is not registered. Please register the patient first.")



def doctor_patient_communication_interface():
    """Doctor-Patient Communication Section with Telegram Messages."""
    st.subheader("Doctor-Patient Communication Section")
    patient_id = st.text_input("Enter Patient ID for Communication")
    message = st.text_area("Enter Message for the Patient")
    appointment_date = st.date_input("Schedule Appointment Date")
    appointment_time = st.time_input("Schedule Appointment Time")

    if st.button("Send Message"):
        if is_patient_registered(patient_id):
            message_text = (
                f"📩 Message from Doctor\n"
                f"Patient ID: {patient_id}\n"
                f"Message: {message}\n"
                f"Appointment Scheduled: {appointment_date} at {appointment_time}"
            )
            send_telegram_message(message_text)
            st.success(f"Message sent to Patient ID: {patient_id} and sent via Telegram.")
        else:
            st.error("Patient is not registered. Please register the patient first.")



# Streamlit App Logic
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "users" not in st.session_state:
    try:
        with open(USER_DB, "r") as f:
            st.session_state["users"] = json.load(f)
    except json.JSONDecodeError:
        st.session_state["users"] = {}

# Main application entry point
if not st.session_state["logged_in"]:
    page = st.selectbox("Choose an option", ["Login", "Create Account"])
    if page == "Login":
        login_page()
    elif page == "Create Account":
        register_page()
else:
    main_interface()
