import os
import requests


CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")

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
            # Update .env file with new tokens
            with open(".env", "w") as env_file:
                env_file.write(f"CLIENT_ID={CLIENT_ID}\n")
                env_file.write(f"CLIENT_SECRET={CLIENT_SECRET}\n")
                env_file.write(f"ACCESS_TOKEN={new_access_token}\n")
                env_file.write(f"REFRESH_TOKEN={new_refresh_token}\n")

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
