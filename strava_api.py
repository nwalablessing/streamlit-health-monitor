import os
import requests
import streamlit as st  # ✅ Use Streamlit Secrets (recommended)

# ✅ Use Streamlit Secrets if available, otherwise fallback to environment variables
CLIENT_ID = st.secrets["CLIENT_ID"] if "CLIENT_ID" in st.secrets else os.getenv("CLIENT_ID")
CLIENT_SECRET = st.secrets["CLIENT_SECRET"] if "CLIENT_SECRET" in st.secrets else os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = st.secrets["REFRESH_TOKEN"] if "REFRESH_TOKEN" in st.secrets else os.getenv("REFRESH_TOKEN")

def refresh_access_token():
    """Refresh Strava access token if expired."""
    url = "https://www.strava.com/api/v3/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }
    
    response = requests.post(url, data=payload)
    data = response.json()
    
    if response.status_code == 200:
        new_access_token = data.get("access_token")
        new_refresh_token = data.get("refresh_token")

        if new_access_token:
            print("✅ Strava token refreshed successfully!")
            return new_access_token
        else:
            print("❌ Failed to retrieve new access token.")
            return None
    else:
        print(f"❌ Error refreshing token: {data}")
        return None

def fetch_activities():
    """Fetch latest activities from Strava after ensuring valid access token."""
    access_token = refresh_access_token()
    
    if not access_token:
        print("❌ No valid access token available.")
        return None

    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        activities = response.json()
        if activities:
            return activities
        else:
            print("⚠️ No recent activity found in Strava.")
            return None
    else:
        print(f"❌ Failed to fetch activities: {response.json()}")
        return None
