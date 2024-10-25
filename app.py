# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title and description
st.title("Formula 1 Data Analysis and Prediction App")
st.write("Explore Formula 1 data, analyze past race statistics, and predict future race outcomes using machine learning models.")

# Define function to get data from F1 API
@st.cache
def get_f1_data(endpoint):
    base_url = "https://ergast.com/api/f1/"
    url = f"{base_url}{endpoint}.json?limit=1000"
    response = requests.get(url)
    data = response.json()
    return data

# Load driver standings data
st.subheader("Driver Standings Data")
standings_data = get_f1_data("2023/driverStandings")
drivers = standings_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

# Convert driver standings to DataFrame
driver_standings_df = pd.DataFrame([
    {
        "position": driver["position"],
        "driver": f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}",
        "points": driver["points"],
        "wins": driver["wins"],
        "constructor": driver["Constructors"][0]["name"]
    }
    for driver in drivers
])
st.write(driver_standings_df)

# Display basic analysis of drivers' points distribution
st.subheader("Driver Points Distribution")
fig, ax = plt.subplots()
sns.histplot(driver_standings_df['points'].astype(float), kde=True, ax=ax)
st.pyplot(fig)

# Load race results data
st.subheader("Race Results Data")
race_results_data = get_f1_data("2023/results")
races = race_results_data['MRData']['RaceTable']['Races']

# Convert race results to DataFrame
race_results_df = pd.DataFrame([
    {
        "race_name": race["raceName"],
        "round": race["round"],
        "date": race["date"],
        "circuit": race["Circuit"]["circuitName"],
        "driver": f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
        "constructor": result["Constructor"]["name"],
        "position": result["position"],
        "status": result["status"]
    }
    for race in races for result in race["Results"]
])
st.write(race_results_df)

# Get unique driver and constructor lists from raw data for dropdowns
unique_drivers = race_results_df['driver'].unique()
unique_constructors = race_results_df['constructor'].unique()

# Data Preprocessing for ML Model
st.subheader("Data Preprocessing and Model Training")
# Encode categorical variables using mappings
race_results_df['position'] = pd.to_numeric(race_results_df['position'], errors='coerce').fillna(0)
race_results_df['constructor_code'] = race_results_df['constructor'].astype('category').cat.codes
race_results_df['driver_code'] = race_results_df['driver'].astype('category').cat.codes

# Update driver and constructor mappings after encoding
driver_map = dict(enumerate(race_results_df['driver'].astype('category').cat.categories))
constructor_map = dict(enumerate(race_results_df['constructor'].astype('category').cat.categories))

# Select features and target
X = race_results_df[['constructor_code', 'driver_code']]
y = race_results_df['position'] <= 3  # Predicting if the driver finished in the top 3

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")

# Display the confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Top 3', 'Top 3'], yticklabels=['Not Top 3', 'Top 3'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Predict for a new driver and constructor combination
st.subheader("Predict Race Outcome")

# Dropdown with driver and constructor names using unique values directly
constructor_name = st.selectbox("Select Constructor", options=unique_constructors)
driver_name = st.selectbox("Select Driver", options=unique_drivers)

# Map selected names to their encoded codes
constructor_code = race_results_df[race_results_df['constructor'] == constructor_name]['constructor_code'].iloc[0]
driver_code = race_results_df[race_results_df['driver'] == driver_name]['driver_code'].iloc[0]

# Predict the likelihood of finishing in the top 3
prediction = model.predict([[constructor_code, driver_code]])
if prediction:
    st.write(f"{driver_name} driving for {constructor_name} has a high likelihood of finishing in the top 3!")
else:
    st.write(f"{driver_name} driving for {constructor_name} is unlikely to finish in the top 3.")

st.write("Thank you for using the F1 Data Analysis and Prediction App.")
