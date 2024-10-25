import os
import streamlit as st
import fastf1 as ff1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

cache_path = os.path.join(os.getcwd(), 'cache')
os.makedirs(cache_path, exist_ok=True)

# Initialize FastF1 caching
ff1.Cache.enable_cache('cache')

# Page configuration
st.set_page_config(
    page_title="F1 Data Analysis",
    page_icon="🏎️",
    layout="wide"
)

# App title and description
st.title("Formula 1 Data Analysis App 🏎️")
st.markdown("""
Explore and analyze Formula 1 data, including race laps, driver telemetry, and timing data.
This app demonstrates data analysis and visualization skills using real F1 data.
""")

# Sidebar options
st.sidebar.header("Select Data")
year = st.sidebar.selectbox("Select Year", list(range(2018, datetime.now().year + 1)), index=0)
race_round = st.sidebar.selectbox("Select Race Round", list(range(1, 24)))

# Load session data
@st.cache_data
def load_session_data(year, race_round):
    try:
        session = ff1.get_session(year, race_round, 'R')
        session.load()
        return session
    except Exception as e:
        st.error(f"Error loading data for year {year}, round {race_round}: {e}")
        return None

# Fetch and load data based on selections
session = load_session_data(year, race_round)
if session:
    st.subheader(f"Data for {session.event['EventName']} - {session.event['EventDate'].date()}")

    # Show basic race details
    st.markdown(f"""
    - **Location**: {session.event['Location']}
    - **Country**: {session.event['Country']}
    - **Laps**: {session.event['Laps']}
    - **Circuit Length**: {session.event['CircuitLength']}
    """)
    
    # Data analysis options
    analysis_option = st.sidebar.radio(
        "Select Analysis",
        ("Lap Times", "Telemetry Comparison", "Driver Fastest Lap", "Pit Stops Analysis")
    )

    # Lap Times Analysis
    if analysis_option == "Lap Times":
        st.subheader("Lap Times Analysis")
        
        # Fetch lap times data
        laps = session.laps
        driver_lap_times = laps[["Driver", "LapTime", "LapNumber"]]
        
        # Display data
        st.write("Lap times data for all drivers:")
        st.dataframe(driver_lap_times)
        
        # Plot lap times
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=driver_lap_times, x="LapNumber", y="LapTime", hue="Driver")
        plt.title("Lap Times for Each Driver")
        plt.xlabel("Lap Number")
        plt.ylabel("Lap Time (s)")
        plt.xticks(rotation=90)
        st.pyplot(plt)

    # Telemetry Comparison
    elif analysis_option == "Telemetry Comparison":
        st.subheader("Telemetry Comparison")
        
        # Select two drivers to compare telemetry data
        drivers = session.drivers
        driver_1 = st.selectbox("Select Driver 1", drivers)
        driver_2 = st.selectbox("Select Driver 2", drivers)
        
        if driver_1 and driver_2 and driver_1 != driver_2:
            # Get telemetry for fastest laps of each driver
            laps_1 = session.laps.pick_driver(driver_1).pick_fastest()
            laps_2 = session.laps.pick_driver(driver_2).pick_fastest()
            
            telemetry_1 = laps_1.get_car_data().add_distance()
            telemetry_2 = laps_2.get_car_data().add_distance()
            
            # Plot telemetry comparison
            fig, ax = plt.subplots(2, 1, figsize=(14, 8))
            ax[0].plot(telemetry_1['Distance'], telemetry_1['Speed'], label=f"{driver_1}", color="blue")
            ax[0].plot(telemetry_2['Distance'], telemetry_2['Speed'], label=f"{driver_2}", color="red")
            ax[0].set_title("Speed Comparison")
            ax[0].set_xlabel("Distance (m)")
            ax[0].set_ylabel("Speed (km/h)")
            ax[0].legend()
            
            ax[1].plot(telemetry_1['Distance'], telemetry_1['Throttle'], label=f"{driver_1}", color="blue")
            ax[1].plot(telemetry_2['Distance'], telemetry_2['Throttle'], label=f"{driver_2}", color="red")
            ax[1].set_title("Throttle Comparison")
            ax[1].set_xlabel("Distance (m)")
            ax[1].set_ylabel("Throttle (%)")
            ax[1].legend()
            
            st.pyplot(fig)
    
    # Driver Fastest Lap Analysis
    elif analysis_option == "Driver Fastest Lap":
        st.subheader("Driver Fastest Lap")
        
        # Choose a driver
        driver = st.selectbox("Select Driver", session.drivers)
        
        # Get fastest lap for selected driver
        fastest_lap = session.laps.pick_driver(driver).pick_fastest()
        st.write(f"Fastest lap for {driver}: {fastest_lap['LapTime']}")
        
        # Plot telemetry data for the fastest lap
        telemetry = fastest_lap.get_car_data().add_distance()
        plt.figure(figsize=(12, 6))
        plt.plot(telemetry['Distance'], telemetry['Speed'], label="Speed (km/h)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Speed (km/h)")
        plt.title(f"{driver}'s Fastest Lap Speed Profile")
        st.pyplot(plt)
    
    # Pit Stops Analysis
    elif analysis_option == "Pit Stops Analysis":
        st.subheader("Pit Stops Analysis")
        
        # Display pit stops data
        pit_stops = session.laps[session.laps['PitInTime'].notna()]
        st.write("Pit stops for all drivers:")
        st.dataframe(pit_stops[['Driver', 'LapNumber', 'PitInTime', 'PitOutTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']])
        
        # Plot pit stops per driver
        pit_counts = pit_stops.groupby("Driver").size().sort_values()
        plt.figure(figsize=(10, 6))
        pit_counts.plot(kind="barh", color="orange")
        plt.title("Number of Pit Stops per Driver")
        plt.xlabel("Number of Pit Stops")
        st.pyplot(plt)

# App footer
st.markdown("---")
st.markdown("Created by [Your Name](https://www.linkedin.com/in/yourprofile)")
