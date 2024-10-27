import streamlit as st
import fastf1 as ff1
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from fastf1 import plotting

# Enable caching for FastF1 data to improve performance
ff1.Cache.enable_cache('./cache')  

# Set up Streamlit page
st.title("F1 Data Insights - AI-Powered Product Manager Portfolio")
st.markdown("""
    This app showcases Formula 1 data using the FastF1 library, demonstrating data analysis skills for an AI-powered product manager.
""")

# Select Season and Race Options
st.sidebar.header("F1 Data Explorer")
year = st.sidebar.selectbox("Select Season", range(2023, 2018, -1))
event_schedule = ff1.get_event_schedule(year)

# Display available races in the selected season
race_names = event_schedule['EventName'].tolist()
race = st.sidebar.selectbox("Select Race", race_names)

# Load Session Data
event = event_schedule[event_schedule['EventName'] == race].iloc[0]
session = st.sidebar.selectbox("Select Session", ["FP1", "FP2", "FP3", "Q", "R"])
f1_session = ff1.get_session(year, event['EventName'], session)
f1_session.load()

# Display event and session details
st.subheader(f"{race} - {session}")
st.write(f"Location: {event['Country']}, Date: {event['EventDate']}")

# Driver Analysis
driver = st.sidebar.selectbox("Select Driver", f1_session.drivers)
driver_laps = f1_session.laps.pick_driver(driver)  # Filter only selected driver's laps
st.subheader(f"Driver: {driver} - Analysis")

# Plotting Telemetry Data for a specific lap
st.markdown("### Lap Telemetry Data")
lap_number = st.sidebar.selectbox("Select Lap", driver_laps['LapNumber'])
selected_lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]  # Ensure selecting only one lap
lap_telemetry = selected_lap.get_car_data().add_distance()  # Get telemetry for the selected lap

# Plot telemetry
fig, ax = plt.subplots()
ax.plot(lap_telemetry['Distance'], lap_telemetry['Speed'], label="Speed")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Speed (km/h)")
ax.set_title(f"Telemetry Data for Lap {lap_number}")
st.pyplot(fig)

# Display Sector Times
st.markdown("### Sector Times")
sector_times = driver_laps[['LapNumber', 'Sector1Time', 'Sector2Time', 'Sector3Time']].dropna()
st.dataframe(sector_times)

# Plot Cumulative Lap Times (Race Analysis)
if session == "R":
    st.markdown("### Cumulative Lap Times for Race Session")
    cumulative_times = driver_laps[['LapNumber', 'LapTime']].dropna()
    fig_cum = px.line(cumulative_times, x="LapNumber", y="LapTime", title=f"{driver} Cumulative Lap Times")
    st.plotly_chart(fig_cum)

# Fastest Laps of All Drivers
st.markdown("### Fastest Laps of All Drivers")
fastest_laps = f1_session.laps.pick_quicklaps()
fastest_laps_summary = fastest_laps[['Driver', 'LapTime']].sort_values('LapTime')
st.dataframe(fastest_laps_summary)

# Display interactive session-level insights and other key metrics
st.sidebar.header("Additional Analysis")
if st.sidebar.checkbox("Show Top Speed"):
    top_speeds = f1_session.laps.pick_quicklaps().get_car_data().Speed.max()
    st.write(f"Top Speed for {session}: {top_speeds:.2f} km/h")

# Session Summary
st.markdown("### Session Summary")
st.write(f1_session.results[['Abbreviation', 'Position', 'Points', 'Time']])