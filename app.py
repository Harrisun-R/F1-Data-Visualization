# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title and description
st.title("Advanced Formula 1 Data Analysis and Prediction App")
st.write("Explore Formula 1 data, analyze past race statistics, and predict future race outcomes using multiple machine learning models.")

# Define function to get data from F1 API
@st.cache
def get_f1_data(endpoint):
    base_url = "https://ergast.com/api/f1/"
    url = f"{base_url}{endpoint}.json?limit=1000"
    response = requests.get(url)
    data = response.json()
    return data

# Load driver standings data for 2023
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

# Load race results data and allow user to select a year
year = st.selectbox("Select Year", options=[str(y) for y in range(2010, 2024)], index=13)
race_results_data = get_f1_data(f"{year}/results")
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

# Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis")

# Points distribution by constructor
st.subheader("Points Distribution by Constructor")
constructor_points_df = driver_standings_df.groupby("constructor")["points"].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(x="points", y="constructor", data=constructor_points_df, ax=ax)
ax.set_title("Total Points by Constructor in 2023")
st.pyplot(fig)

# Wins distribution by driver
st.subheader("Wins Distribution by Driver")
driver_wins_df = driver_standings_df[["driver", "wins"]].sort_values(by="wins", ascending=False)
fig, ax = plt.subplots()
sns.barplot(x="wins", y="driver", data=driver_wins_df, ax=ax)
ax.set_title("Total Wins by Driver in 2023")
st.pyplot(fig)

# Data Preprocessing for Machine Learning Models
st.subheader("Data Preprocessing and Model Selection")
race_results_df['position'] = pd.to_numeric(race_results_df['position'], errors='coerce').fillna(0)
race_results_df['constructor_code'] = race_results_df['constructor'].astype('category').cat.codes
race_results_df['driver_code'] = race_results_df['driver'].astype('category').cat.codes

# Select features and target
X = race_results_df[['constructor_code', 'driver_code']]
y = race_results_df['position'] <= 3  # Predicting if the driver finished in the top 3

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model_option = st.selectbox("Select a Machine Learning Model", ("Random Forest", "Gradient Boosting", "Logistic Regression"))

if model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_option == "Gradient Boosting":
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
else:
    model = LogisticRegression(max_iter=1000)

# Train the model and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Top 3', 'Top 3'], yticklabels=['Not Top 3', 'Top 3'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Feature Importance (for tree-based models)
if model_option in ["Random Forest", "Gradient Boosting"]:
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    feature_importances.plot(kind='barh', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Predict for a new driver and constructor combination
st.subheader("Predict Race Outcome for Selected Driver and Constructor")
unique_drivers = race_results_df['driver'].unique()
unique_constructors = race_results_df['constructor'].unique()
constructor_name = st.selectbox("Select Constructor", options=unique_constructors)
driver_name = st.selectbox("Select Driver", options=unique_drivers)

# Map selected names to encoded values
constructor_code = race_results_df[race_results_df['constructor'] == constructor_name]['constructor_code'].iloc[0]
driver_code = race_results_df[race_results_df['driver'] == driver_name]['driver_code'].iloc[0]

# Predict outcome
prediction = model.predict([[constructor_code, driver_code]])
if prediction:
    st.write(f"{driver_name} driving for {constructor_name} has a high likelihood of finishing in the top 3!")
else:
    st.write(f"{driver_name} driving for {constructor_name} is unlikely to finish in the top 3.")

st.write("Thank you for using the Advanced F1 Data Analysis and Prediction App.")
