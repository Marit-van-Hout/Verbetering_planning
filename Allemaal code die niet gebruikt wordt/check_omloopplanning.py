import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from matplotlib.patches import Patch

def check_batterij_status(uploaded_file, distance_matrix, start_batterij=270, min_batterij=30):
    errors = []
    
    # Gegevens inladen en DataFrame samenvoegen
    df = pd.merge(uploaded_file, distance_matrix, on=['startlocatie', 'eindlocatie', 'buslijn'], how='left')

    # Consumptie voor kilometers
    consumption_per_km = (0.7 + 2.5) / 2  
    df['consumptie_kWh'] = (df['afstand in meters'] / 1000) * consumption_per_km

    # Consumptie voor idle activiteiten
    df.loc[df['activiteit'] == 'idle', 'consumptie_kWh'] = 0.01

    # Laadsnelheden instellen
    charging_speed_90 = 450 / 60  # kWh per minuut voor opladen tot 90%
    charging_speed_10 = 60 / 60   # kWh per minuut voor opladen van 90% tot 100%

    # Beginwaarden
    battery_level = start_batterij
    vorig_omloopnummer = df['omloop nummer'].iloc[0]

    # Itereren door de DataFrame
    for i, row in df.iterrows():
        # Controleer of het een nieuwe omloop is
        if row['omloop nummer'] != vorig_omloopnummer:
            # Energieverbruik afhalen vóór het resetten van de batterij
            battery_level -= row['consumptie_kWh']
            
            # Reset de batterij naar start_batterij
            battery_level = start_batterij

        # Opladen
        if row['activiteit'] == 'opladen':
            # Start- en eindtijd ophalen en de duur berekenen
            start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            charging_duration = (end_time - start_time).total_seconds() / 60

            # Bepaal de laadsnelheid
            if battery_level <= 243:
                charge_power = charging_speed_90 * charging_duration
            else:
                charge_power = charging_speed_10 * charging_duration

            # Opladen en aanpassen van de batterijstatus
            battery_level += charge_power

        else:
            # Verminderen met de consumptie
            battery_level -= row['consumptie_kWh']

        # Controleer of de batterijstatus onder het minimum komt
        if battery_level < min_batterij:
            warning_message = f"Warning: Battery under {min_batterij} kWh at bus {row['omloop nummer']} on {row['starttijd']}"
            st.error(warning_message)
            errors.append(warning_message)

        # Bij nieuwe omloop het omloopnummer updaten
        vorig_omloopnummer = row['omloop nummer']
    
    return errors

def check_route_continuity(bus_planning):
        """ Check if the endpoint of route n matches the start point of route n+1.
        Parameters:
            - bus_planning: DataFrame with route data.
        Output: Print messages if there are inconsistencies.
        """
    
        #Controleer op NaN-waarden in 'omloop nummer'
        if bus_planning['omloop nummer'].isna().any():
            st.error("NaN values found in 'omloop nummer' column.")
            return False

    # Controleer de continuïteit van de routes
        for i in range(len(bus_planning) - 1):
            current_end_location = bus_planning.at[i, 'eindlocatie']
            next_start_location = bus_planning.at[i + 1, 'startlocatie']
            omloop_nummer = bus_planning.at[i, 'omloop nummer']
            next_start_time = bus_planning.at[i + 1, 'starttijd'].time() # Haal de starttijd van de volgende route op

            if current_end_location != next_start_location:
                st.error(f"Route continuity issue between bus number {omloop_nummer:.0f} at {next_start_time}: "
                        f"ends at {current_end_location} and next route starts at {next_start_location}.")

def driven_rides(bus_planning):
    clean_bus_planning = bus_planning[['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']]
    clean_bus_planning = clean_bus_planning.dropna(subset=['buslijn']) 
    return clean_bus_planning

def every_ride_covered(bus_planning, time_table):
    errors = []
    bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], errors='coerce')
    time_table['starttijd'] = pd.to_datetime(time_table['starttijd'], errors='coerce')
    time_table = time_table.rename(columns={'vertrektijd': 'starttijd'})

    bus_planning_sorted = bus_planning.sort_values(by=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']).reset_index(drop=True)
    time_table_sorted = time_table.sort_values(by=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']).reset_index(drop=True)

    difference_bus_planning_to_time_table = bus_planning_sorted.merge(
        time_table_sorted, on=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn'], how='outer', indicator=True
    ).query('_merge == "left_only"')

    difference_time_table_to_bus_planning = bus_planning_sorted.merge(
        time_table_sorted, on=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn'], how='outer', indicator=True
    ).query('_merge == "right_only"')

    if not difference_bus_planning_to_time_table.empty:
        errors.append('Rows only contained in bus planning:\n', difference_bus_planning_to_time_table)
        st.dataframe(difference_bus_planning_to_time_table)
        return False, errors
        
    if not difference_time_table_to_bus_planning.empty:
        errors.append('Rows only contained in time table:\n', difference_time_table_to_bus_planning)
        st.dataframe(difference_time_table_to_bus_planning)
        return False, errors
        
    if difference_bus_planning_to_time_table.empty and difference_time_table_to_bus_planning.empty:
        return 'Bus planning is equal to time table'
    
    return True, errors

def check_travel_time(bus_planning, distance_matrix):
    errors = []
    bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], format='%H:%M:%S', errors='coerce')
    bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], format='%H:%M:%S', errors='coerce')

    bus_planning['verschil_in_minuten'] = (bus_planning['eindtijd'] - bus_planning['starttijd']).dt.total_seconds() / 60

    merged_df = pd.merge(
        bus_planning,
        distance_matrix,
        on=['startlocatie', 'eindlocatie', 'buslijn'],
        how='inner'  
    )

    for index, row in merged_df.iterrows():
        if not (row['min reistijd in min'] <= row['verschil_in_minuten'] <= row['max reistijd in min']):
            errors.append(f'Row {index}: The difference in minutes ({row["verschil_in_minuten"]:.0f}) is not between {row["max reistijd in min"]} and {row["min reistijd in min"]} for bus route {row["buslijn"]} from {row["startlocatie"]} to {row["eindlocatie"]}.')
            st.error(errors[-1])
            return False, errors
            
    return True, errors

def plot_schedule_from_excel(bus_planning):
    """Plot een Gantt-grafiek voor busplanning op basis van een DataFrame."""

    # Controleer of de vereiste kolommen aanwezig zijn
    required_columns = ['starttijd', 'eindtijd', 'buslijn', 'omloop nummer', 'activiteit']
    if not all(col in bus_planning.columns for col in required_columns):
        st.error("Een of meer vereiste kolommen ontbreken in de busplanning.")
        return

    bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], errors='coerce')
    bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], errors='coerce')

    # Verwijder rijen met NaT in starttijd of eindtijd
    bus_planning = bus_planning.dropna(subset=['starttijd', 'eindtijd'])

    bus_planning['duration'] = (bus_planning['eindtijd'] - bus_planning['starttijd']).dt.total_seconds() / 3600

    min_duration = 0.05  
    bus_planning['duration'] = bus_planning['duration'].apply(lambda x: max(x, min_duration))

    color_map = {
        '400.0': 'blue',
        '401.0': 'yellow',
        'materiaal rit': 'green',
        'idle': 'red',
        'opladen': 'orange'
    }

    bus_planning['buslijn'] = bus_planning['buslijn'].astype(str)

    def determine_color(row):
        if pd.notna(row['buslijn']) and row['buslijn'] in color_map:
            return color_map[row['buslijn']]  
        elif row['activiteit'] in color_map:
            return color_map[row['activiteit']]  
        else:
            return 'gray' 

    bus_planning['color'] = bus_planning.apply(determine_color, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    omloopnummers = bus_planning['omloop nummer'].unique()
    omloop_indices = {omloop: i for i, omloop in enumerate(omloopnummers)}

    for omloop in omloopnummers:
        trips = bus_planning[bus_planning['omloop nummer'] == omloop]

        if trips.empty:
            ax.barh(omloop_indices[omloop], 1, left=0, color='black', edgecolor='black')
            continue

        for _, trip in trips.iterrows():
            starttime = trip['starttijd']
            duration = trip['duration']
            color = trip['color'] 

            ax.barh(omloop_indices[omloop], duration, left=starttime.hour + starttime.minute / 60,
                    color=color, edgecolor='black')

    ax.set_yticks(list(omloop_indices.values()))
    ax.set_yticklabels(list(omloop_indices.keys()))

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Bus Number')
    ax.set_title('Gantt Chart for Bus Scheduling')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Regular trip 400'),
        Patch(facecolor='yellow', edgecolor='black', label='Regular trip 401'),
        Patch(facecolor='green', edgecolor='black', label='Deadhead trip'),
        Patch(facecolor='red', edgecolor='black', label='Idle'),
        Patch(facecolor='orange', edgecolor='black', label='Charging')
    ]
 
    ax.legend(handles=legend_elements, title='Legenda')

    st.pyplot(fig)

st.image("logo_transdev_klein.png", width=200)
st.sidebar.title('Navigation')

# Streamlit 
def bus_checker_page(): 
    st.title("Bus Planning Checker")

    uploaded_file = st.file_uploader("Upload omloopplanning.xlsx", type="xlsx")
    given_data = st.file_uploader("Upload Connexxion data - 2024-2025.xlsx", type="xlsx")

    if uploaded_file and given_data:
        with st.spinner('Data is being processed'): 
            try:
                bus_planning = pd.read_excel(uploaded_file)
                time_table = pd.read_excel(given_data, sheet_name='Dienstregeling')
                distance_matrix = pd.read_excel(given_data, sheet_name="Afstandsmatrix")
            except Exception as e:
                st.error(f"Error reading Excel files: {str(e)}")
                return

            st.write('Your Bus Planning:')
            st.dataframe(bus_planning)

            st.write('Gantt Chart Bus Planning')
            plot_schedule_from_excel(bus_planning)  # Hier gebruiken we bus_planning in plaats van uploaded_file

            st.write('Errors in Planning')
            errors = []

            if bus_planning.empty or time_table.empty or distance_matrix.empty:
                st.error("One or more DataFrames are empty. Please check the uploaded files.")
                return

            try: 
                errors += check_batterij_status(bus_planning, distance_matrix)
            except Exception as e:
                errors.append(f'Something went wrong checking battery: {str(e)}')

            try:
                errors += check_route_continuity(bus_planning) 
            except Exception as e:
                errors.append(f'Something went wrong checking route continuity: {str(e)}')

            try:
                bus_planning = driven_rides(bus_planning)
            except Exception as e:
                errors.append(f'Something went wrong checking driven rides: {str(e)}')

            try:
                errors += every_ride_covered(bus_planning, time_table)  # Corrected from distance_matrix to time_table
            except Exception as e:
                errors.append(f'Something went wrong checking if each ride is covered: {str(e)}')

            try:
                errors += check_travel_time(bus_planning, distance_matrix)
            except Exception as e:
                errors.append(f'Something went wrong checking the travel time: {str(e)}')

            if errors:
                st.write("Errors gevonden:")
                for error in errors:
                    st.write(error)
            else:
                st.success("Schedule is valid!")


def how_it_works_page():
    st.title("How It Works")
    st.write("This is how the **Bus Planning Checker** works. Here's the explanation of how everything functions.")

    st.write("The app checks the following conditions:")

    st.markdown("""
    1. **Battery Status Check**: Ensures the battery status is not under **10%** of the State of Health, which is **30 kWh**.
    2. **Route Endpoint Check**: Verifies if the endpoint of route **n** matches the start point of the next route.
    3. **Travel Time Check**: Confirms that the actual travel time is within the specified minimum and maximum travel time.
    """)

def help_page():
    st.title('Help')
    st.write('This page gives help to the people who need it.')

# Hoofd pagina selector
page = st.sidebar.selectbox('Select a Page', ['Bus Planning Checker', 'How It Works', 'Help'])

if page == 'Bus Planning Checker':
    bus_checker_page()
elif page == 'How It Works':
    how_it_works_page()
elif page == 'Help':
    help_page()
