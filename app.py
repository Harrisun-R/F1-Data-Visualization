import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="F1 Data Visualization", layout="wide")

# Function to fetch data from OpenF1 API
def fetch_openf1_data(endpoint):
    base_url = f"https://api.openf1.org/v1/{endpoint}"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return None

# BASIC FUNCTIONALITY
def basic_version():
    st.header("Basic F1 Data Visualization")

    # Driver Standings for a Specific Season
    year = st.selectbox("Select Season Year", [2023, 2022, 2021, 2020])
    data = fetch_openf1_data(f"driver-standings/{year}")
    
    if data:
        standings = pd.DataFrame(data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings'])
        st.subheader(f"Driver Standings for {year}")
        st.dataframe(standings[['position', 'Driver', 'points', 'wins']])

        # Bar chart: Standings
        fig = px.bar(standings, x='Driver', y='points', title=f"Driver Points for {year}")
        st.plotly_chart(fig)

    # Qualifying Results
    round_ = st.selectbox("Select Race Round for Qualifying", range(1, 23))
    qual_data = fetch_openf1_data(f"qualifying/{year}/{round_}")
    
    if qual_data:
        qual_results = pd.DataFrame(qual_data['MRData']['RaceTable']['Races'][0]['QualifyingResults'])
        st.subheader(f"Qualifying Results for Round {round_} in {year}")
        st.dataframe(qual_results[['Driver', 'position', 'Q1', 'Q2', 'Q3']])
        
        # Starting Position Chart
        fig = px.bar(qual_results, x='Driver', y='position', title="Qualifying Starting Positions")
        st.plotly_chart(fig)
    
    # Circuit Information
    circuit_data = fetch_openf1_data(f"circuits/{round_}/{year}")
    if circuit_data:
        circuit_info = circuit_data['MRData']['CircuitTable']['Circuits'][0]
        st.subheader(f"Circuit Information for Round {round_} in {year}")
        st.write(f"**Circuit Name:** {circuit_info['circuitName']}")
        st.write(f"**Location:** {circuit_info['Location']['locality']}, {circuit_info['Location']['country']}")
        st.write(f"**Track Length:** {circuit_info.get('length', 'N/A')}")

# ADVANCED FUNCTIONALITY
def advanced_version():
    st.header("Advanced F1 Data Visualization")

    # Historical Race Comparison
    def historical_race_comparison():
        years = [2023, 2022, 2021, 2020]
        races = list(range(1, 23))
        selected_years = st.multiselect("Select Race Years to Compare", years, default=[2023, 2022])
        selected_round = st.selectbox("Select Race Round", races)
        
        data_frames = []
        for year in selected_years:
            data = fetch_openf1_data(f"race-results/{year}/{selected_round}")
            if data:
                df = pd.DataFrame(data['MRData']['RaceTable']['Races'][0]['Results'])
                df['Year'] = year
                data_frames.append(df[['Year', 'position', 'Driver', 'Constructor', 'Time']])

        if data_frames:
            combined_df = pd.concat(data_frames)
            st.subheader(f"Race Comparison for Round {selected_round}")
            st.dataframe(combined_df)

            # Grouped bar chart: Position Comparison Across Years
            fig = px.bar(combined_df, x='position', y='Time', color='Year', barmode='group', title=f"Position Comparison for Round {selected_round}")
            st.plotly_chart(fig, use_container_width=True)

    # Predictive Analytics for Driver/Constructor Standings
    def predictive_analytics():
        data = fetch_openf1_data("driver-standings")
        if data:
            df = pd.DataFrame(data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings'])
            df['Round'] = df['position'].astype(int)  # Assume position correlates with rounds for simplicity
            X = np.array(df['Round']).reshape(-1, 1)
            y = df['points'].astype(float)
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_rounds = np.array([[i] for i in range(df['Round'].max() + 1, df['Round'].max() + 6)])  # Predict next 5 rounds
            predictions = model.predict(future_rounds)
            
            prediction_df = pd.DataFrame({'Round': future_rounds.flatten(), 'Predicted Points': predictions})
            st.subheader("Predicted Driver Standings for Upcoming Rounds")
            st.dataframe(prediction_df)

            fig = px.line(df, x='Round', y='points', title="Driver Points Prediction vs Actual", labels={'Round': 'Race Rounds', 'points': 'Points'})
            fig.add_scatter(x=prediction_df['Round'], y=prediction_df['Predicted Points'], mode='lines+markers', name='Predicted Points')
            st.plotly_chart(fig, use_container_width=True)

    # Weather Conditions Analysis
    def weather_conditions_analysis():
        st.subheader("Weather Conditions Analysis")
        year = st.selectbox("Select Year for Weather Data", [2023, 2022, 2021])
        round_ = st.selectbox("Select Race Round", range(1, 23))
        
        weather_data = fetch_openf1_data(f"weather/{year}/{round_}")
        if weather_data:
            weather_df = pd.DataFrame(weather_data['MRData']['RaceTable']['Races'][0]['Weather'])
            st.dataframe(weather_df)
            
            fig = px.line(weather_df, x='time', y='temperature', title="Temperature Changes During the Race")
            fig.add_scatter(x=weather_df['time'], y=weather_df['rain'], mode='lines', name='Rain Level')
            st.plotly_chart(fig, use_container_width=True)

    # Team Consistency Over Season
    def team_consistency_over_season():
        st.subheader("Team Consistency Over the Season")
        selected_year = st.selectbox("Select Season Year", [2023, 2022, 2021])
        
        data = fetch_openf1_data(f"team-consistency/{selected_year}")
        if data:
            team_df = pd.DataFrame(data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings'])
            team_df['Race Count'] = team_df.groupby('Constructor')['points'].transform('count')
            team_df['Average Points'] = team_df.groupby('Constructor')['points'].transform('mean')
            st.dataframe(team_df[['Constructor', 'points', 'Race Count', 'Average Points']])
            
            fig = px.bar(team_df, x='Constructor', y='Average Points', title="Average Points Per Race for Each Team")
            st.plotly_chart(fig, use_container_width=True)

    # Advanced functionality options
    option = st.selectbox("Select Advanced Feature", [
        "Historical Race Comparison", 
        "Predictive Analytics for Standings", 
        "Weather Conditions Analysis", 
        "Team Consistency Over Season"
    ])

    if option == "Historical Race Comparison":
        historical_race_comparison()
    elif option == "Predictive Analytics for Standings":
        predictive_analytics()
    elif option == "Weather Conditions Analysis":
        weather_conditions_analysis()
    elif option == "Team Consistency Over Season":
        team_consistency_over_season()

# Sidebar to switch between basic and advanced modes
mode = st.sidebar.selectbox("Choose App Mode", ["Basic", "Advanced"])

# Render the appropriate version based on user selection
if mode == "Basic":
    basic_version()
elif mode == "Advanced":
    advanced_version()

st.sidebar.info("Data is fetched using OpenF1 API and visualized using Plotly and Seaborn.")
st.write("---")
st.write("Created by [Your Name] | [LinkedIn](https://www.linkedin.com/in/yourprofile)")
