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
import xml.etree.ElementTree as ET
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


PATIENT_DB = "patient_data.json"

def save_patient_data(patient_id, patient_name, chat_id):
    """Save registered patient details including Telegram Chat ID."""

    # ✅ Ensure the file exists and is not empty
    if not os.path.exists(PATIENT_DB):
        with open(PATIENT_DB, "w") as f:
            json.dump({}, f)

    try:
        # ✅ Read existing patient data safely
        with open(PATIENT_DB, "r") as f:
            try:
                patients = json.load(f)
                if not isinstance(patients, dict):
                    patients = {}  # Reset if not a dictionary
            except json.JSONDecodeError:
                patients = {}  # Reset if file is corrupt
        
        # ✅ Update the patient info
        patients[patient_id] = {"name": patient_name, "chat_id": chat_id}

        # ✅ Write back to the file safely
        with open(PATIENT_DB, "w") as f:
            json.dump(patients, f, indent=4)

        print(f"✅ Patient {patient_name} (ID: {patient_id}) registered successfully!")

    except OSError as e:
        print(f"❌ Error saving patient data: {e}")
        
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
    chat_id = st.text_input("Enter Patient's Telegram Chat ID")  # ✅ Added this line

    if st.button("Register Patient"):
        if patient_id and patient_name and chat_id:  # ✅ Ensure all fields are filled
            save_patient_data(patient_id, patient_name, chat_id)  # ✅ Now passing chat_id
            st.success(f"Patient {patient_name} with ID {patient_id} has been registered.")
        else:
            st.error("⚠️ Please provide Patient ID, Name, and Chat ID.")


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
            
def parse_health_data(xml_file):
    """Parse export.xml and extract HeartRate, SpO2, BodyMass."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    records = []

    for record in root.findall("Record"):
        rtype = record.attrib.get("type")
        if rtype in [
            "HKQuantityTypeIdentifierHeartRate",
            "HKQuantityTypeIdentifierOxygenSaturation",
            "HKQuantityTypeIdentifierBodyMass"
        ]:
            records.append({
                "type": rtype.split("Identifier")[-1],  # Just get "HeartRate"
                "value": float(record.attrib.get("value", 0)),
                "unit": record.attrib.get("unit"),
                "start_date": record.attrib.get("startDate"),
            })

    df = pd.DataFrame(records)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

# Run this only once to convert XML to CSV and JSON
if __name__ == "__main__":
    df = parse_health_data("export.xml")  # ✅ Put your Apple Health export here
    df.to_csv("apple_health.csv", index=False)
    df.to_json("apple_health.json", orient="records", indent=2)
    print("✅ export.xml converted to apple_health.csv and apple_health.json")

def real_time_monitoring_interface():
    """Real-Time Patient Monitoring using Strava + Apple Health data."""
    st.subheader("📡 Real-Time Patient Monitoring")

    patient_id = st.text_input("Enter Patient ID to Monitor")

    if not is_patient_registered(patient_id):
        st.warning("⚠️ Patient not registered. Please register first.")
        return

    # ========================
    # 🚴 STRAVA: Real-Time Data
    # ========================
    st.markdown("## 🚴 Strava Activity Data")

    if st.button("Fetch Strava Activities"):
        st.info("🔄 Fetching from Strava...")

        activities = fetch_activities()
        if activities and len(activities) > 0:
            df_strava = pd.DataFrame(activities)

            columns = ["name", "type", "distance", "moving_time", "average_speed", "start_date_local", "average_heartrate"]
            df_strava = df_strava[[col for col in columns if col in df_strava.columns]]
            df_strava["start_date_local"] = pd.to_datetime(df_strava["start_date_local"])

            st.success("✅ Strava data loaded")
            st.dataframe(df_strava)

            # Distance plot
            st.write("### 📈 Distance Over Time")
            fig, ax = plt.subplots()
            ax.plot(df_strava["start_date_local"], df_strava["distance"], marker="o", color="blue")
            ax.set_xlabel("Date")
            ax.set_ylabel("Meters")
            ax.set_title("Distance Over Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()
            st.pyplot(fig)

            # Speed plot
            st.write("### ⚡ Speed Over Time")
            fig, ax = plt.subplots()
            ax.plot(df_strava["start_date_local"], df_strava["average_speed"], marker="o", color="red")
            ax.set_xlabel("Date")
            ax.set_ylabel("m/s")
            ax.set_title("Speed Over Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()
            st.pyplot(fig)

            # Heart Rate
            if "average_heartrate" in df_strava.columns and df_strava["average_heartrate"].notnull().any():
                st.write("### ❤️ Heart Rate from Strava")
                fig, ax = plt.subplots()
                ax.plot(df_strava["start_date_local"], df_strava["average_heartrate"], marker="o", color="green")
                ax.set_title("Heart Rate from Strava")
                ax.set_ylabel("bpm")
                ax.set_xlabel("Date")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig.autofmt_xdate()
                st.pyplot(fig)
        else:
            st.error("❌ No activity data retrieved from Strava.")

    # ==========================
    # 🍎 APPLE HEALTH CSV/JSON
    # ==========================
    st.markdown("## 🍎 Apple Health Data")

    format_choice = st.radio("Select Health Data Source", ["CSV", "JSON"])

    df_health = pd.DataFrame()

    try:
        if format_choice == "CSV":
            df_health = pd.read_csv("apple_health.csv")
        elif format_choice == "JSON":
            df_health = pd.read_json("apple_health.json")

        df_health["start_date"] = pd.to_datetime(df_health["start_date"])
        df_health["value"] = pd.to_numeric(df_health["value"], errors="coerce")

        st.success("✅ Apple Health data loaded.")
        st.write("### 🧾 Apple Health Records")
        st.dataframe(df_health.sort_values("start_date", ascending=False))

        for metric in df_health["type"].unique():
            metric_df = df_health[df_health["type"] == metric]
            if not metric_df.empty:
                st.write(f"### 📈 {metric} Over Time")
                fig, ax = plt.subplots()
                ax.plot(metric_df["start_date"], metric_df["value"], marker="o", linestyle="-")
                ax.set_title(f"{metric} Trends")
                ax.set_ylabel(metric_df['unit'].iloc[0])
                ax.set_xlabel("Date")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig.autofmt_xdate()
                st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error loading Apple Health data: {e}")


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
