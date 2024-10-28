import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from matplotlib.patches import Patch

# Load data
#uploaded_file = pd.read_excel('omloopplanning.xlsx')
#bus_planning_2 = pd.read_excel('omloopplanning.xlsx')  
# voor Fleur:
# streamlit run "c:/Users/Fleur/Documents/Visual Studio Code/Verbetering_planning/streamlit_app.py"

def validate_schedule(bus_planning, time_table, distance_matrix):
    """
    Valideert het schema van een busplanning.

    Parameters:
    - row: Een dictionary met de gegevens van de rit.
    - time_table: Een DataFrame met vertrektijden en andere relevante informatie.
    - uploaded_file, actual_capacity, start_times, end_time: Vereiste parameters voor de simulatie en batterijfuncties.
    - distance: Afstand voor batterijverbruik.
    - bus_planning: De planning van de bus.
    - scheduled_orders: Geplande orders voor plotting.
    - distance_matrix: Matrix voor reistijdcontrole.

    Returns:
    - Een lijst van foutmeldingen die zijn opgetreden tijdens de validatie.
    """

    errors = []

    # Parameters
    max_capacity = 300 # maximum capacity in kWh
    SOH = 0.9 # State of Health
    charging_speed_90 = 450 / 60 # kWh per minute when charging to 90%
    actual_capacity = max_capacity * SOH
    consumption_per_km = (0.7 + 2.5) / 2 # kWh per km
    min_idle_time = 15
    # charging_time_10 = 60 / 60 # kWh per minute when charging from 90% to 100% (wordt niet gebruikt)
    # daytime_limit = actual_capacity * 0.9 (wordt niet gebruikt)
    
    def check_batterij_status(uploaded_file, distance_matrix, start_batterij=270, min_batterij=30):
        # Reset indices om ervoor te zorgen dat de dataframes correct worden gestapeld naast elkaar
        uploaded_file = uploaded_file.reset_index(drop=True)
        # Normaliseer kolomnamen
        uploaded_file.columns = uploaded_file.columns.str.strip().str.lower()
        distance_matrix.columns = distance_matrix.columns.str.strip().str.lower()

        # Controleer of 'afstand in meters' kolom aanwezig is in distance_matrix
        # **Controle toegevoegd om te bevestigen dat de kolom bestaat**
        if 'afstand in meters' not in distance_matrix.columns:
            st.error("Kolom 'afstand in meters' ontbreekt in de distance_matrix.")
            errors.append("Kolom 'afstand in meters' ontbreekt in de distance_matrix.")
            return errors

        # Reset de index van de distance_matrix en selecteer de relevante kolommen
        distance_matrix = distance_matrix[['startlocatie', 'eindlocatie', 'buslijn', 'afstand in meters']].reset_index(drop=True)

        # Merge de DataFrames op relevante kolommen
        # **Merge in plaats van concat, zodat afstand in meters wordt meegenomen**
        df = uploaded_file.merge(distance_matrix, on=['startlocatie', 'eindlocatie', 'buslijn'], how='inner')

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
                st.error(f"Waarschuwing: Batterij onder {min_batterij} kWh bij omloop {row['omloop nummer']} op tijd {row['starttijd']}")
                errors.append(f"Waarschuwing: Batterij onder {min_batterij} kWh bij omloop {row['omloop nummer']} op tijd {row['starttijd']}")

            # Bij nieuwe omloop het omloopnummer updaten
            vorig_omloopnummer = row['omloop nummer']
        
        return errors

    # Something went wrong checking route continuity: 'omloop nummer'
    def check_route_continuity(bus_planning):
        """ Check if the endpoint of route n matches the start point of route n+1.
        Parameters:
            - bus_planning: DataFrame with route data.
        Output: Print messages if there are inconsistencies.
        """
    
      # Controleer op NaN-waarden in 'omloop nummer'
        if bus_planning['omloop nummer'].isna().any():
            st.error("NaN values found in 'omloop nummer' column.")
            errors.append("NaN values found in 'omloop nummer' column.")
            return False, errors

        # Controleer de continuïteit van de routes
        for i in range(len(bus_planning) - 1):
            current_end_location = bus_planning.at[i, 'eindlocatie']
            next_start_location = bus_planning.at[i + 1, 'startlocatie']
            omloop_nummer = bus_planning.at[i, 'omloop nummer']
            next_start_time = bus_planning.at[i + 1, 'starttijd'].time() # Haal de starttijd van de volgende route op

            if current_end_location != next_start_location:
                st.error(f"Route continuity issue between bus number {omloop_nummer:.0f} at {next_start_time}: "
                        f"ends at {current_end_location} and next route starts at {next_start_location}.")
                errors.append(f"Route continuity issue between bus number {omloop_nummer:.0f} at {next_start_time}: "
                        f"ends at {current_end_location} and next route starts at {next_start_location}.")
                return False, errors
         
        return True, errors

    def driven_rides(bus_planning):
        """ displays which rides are driven
        Parameters
            omloopplanning: DataFrame
            The full circulation planning data.
        output
            DataFrame
            A cleaned bus planning DataFrame containing only the relevant columns 
            and rows where a bus line is present.
        """
        clean_bus_planning = bus_planning[['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']]
        clean_bus_planning = clean_bus_planning.dropna(subset=['buslijn']) # dropt alle rijen die geen buslijn hebben
        return clean_bus_planning
    
    
    bus_planning = driven_rides(bus_planning)


    def every_ride_covered(bus_planning, time_table):
        """Checks if every ride in the timetable is covered in bus planning.
    
        Parameters: 
            bus_planning : DataFrame
                The DataFrame representing the rides that are actually driven.
            time_table : DataFrame
                The DataFrame representing the rides that are supposed to be driven.
    
        Returns:
            DataFrame or str
                If there are differences, returns a DataFrame with the differences.
                If all rides are covered, returns a success message.
        """
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
            st.error('Rows only contained in bus planning:\n', difference_bus_planning_to_time_table)
            errors.append('Rows only contained in bus planning:\n', difference_bus_planning_to_time_table)
            st.dataframe(difference_bus_planning_to_time_table)
            return False, errors
        
        if not difference_time_table_to_bus_planning.empty:
            st.error('Rows only contained in time table:\n', difference_time_table_to_bus_planning)
            errors.append('Rows only contained in time table:\n', difference_time_table_to_bus_planning)
            st.dataframe(difference_bus_planning_to_time_table)
            return False, errors
        
        if difference_bus_planning_to_time_table.empty and difference_time_table_to_bus_planning.empty:
            return 'Bus planning is equal to time table'
        
        return True, errors
    
  
    def check_travel_time(bus_planning, distance_matrix):
        """
        Check if the time difference between the start time and end time in bus planning
        falls within the minimum and maximum travel time in distance_matrix,
        while start location, end location, and bus line are the same.
    
        Parameters:
        bus_planning : DataFrame
            A DataFrame with columns 'starttijd', 'eindtijd', 'startlocatie', 'eindlocatie', 'buslijn'.
        distance_matrix : DataFrame
            A DataFrame with columns 'startlocatie', 'eindlocatie', 'minimale_reistijd', 'maximale_reistijd', 'buslijn'.
    
        Returns:
        Messages for rows that do not fall within the travel time.
        """
        # Ensure start time and end time are datetime
        bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], format='%H:%M:%S', errors='coerce')
        bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], format='%H:%M:%S', errors='coerce')

        # Calculate the difference in minutes
        bus_planning['verschil_in_minuten'] = (bus_planning['eindtijd'] - bus_planning['starttijd']).dt.total_seconds() / 60

        # Merge both datasets on 'startlocatie', 'eindlocatie' and 'buslijn'
        merged_df = pd.merge(
            bus_planning,
            distance_matrix,
            on=['startlocatie', 'eindlocatie', 'buslijn'],
            how='inner'  # Keep only common rows
        )

        # Check if the difference is within the minimum and maximum travel time
        for index, row in merged_df.iterrows():
            if not (row['min reistijd in min'] <= row['verschil_in_minuten'] <= row['max reistijd in min']):
                st.error(f'Row {index}: The difference in minutes ({row['verschil_in_minuten']:.0f}) is not between {row['max reistijd in min']} and {row['min reistijd in min']} for bus route {row['buslijn']} from {row['startlocatie']} to {row['eindlocatie']}.')
                errors.append(f'Row {index}: The difference in minutes ({row['verschil_in_minuten']:.0f}) is not between {row['max reistijd in min']} and {row['min reistijd in min']} for bus route {row['buslijn']} from {row['startlocatie']} to {row['eindlocatie']}.')
                return False, errors
            
        return True, errors
    # De validatiefuncties aanroepen
    #try:
        #time_table['end_time'] = time_table.apply(calculate_end_time, axis=1)
    #except Exception as e:
        #st.error(f'Something went wrong calculating the end time: {str(e)}')
        
    try: 
        check_batterij_status(bus_planning, distance_matrix, start_batterij=270, min_batterij=30)
    except Exception as e:
        st.error(f'Something went wrong checking battery: {str(e)}')
        errors.append(f'Something went wrong checking battery: {str(e)}')
    
    try:
        check_route_continuity(bus_planning) 
    except Exception as e:
        st.error(f'Something went wrong checking route continuity: {str(e)}')
        errors.append(f'Something went wrong checking route continuity: {str(e)}')
    
    try:
        driven_rides(bus_planning)
    except Exception as e:
        st.error(f'Something went wrong checking driven rides: {str(e)}')
        errors.append(f'Something went wrong checking driven rides: {str(e)}')
    
    try:
        every_ride_covered(bus_planning, time_table)
    except Exception as e:
        st.error(f'Something went wrong checking if each ride is covered: {str(e)}')
        errors.append(f'Something went wrong checking if each ride is covered: {str(e)}')
    
    try:
        check_travel_time(bus_planning, distance_matrix)
    except Exception as e:
        st.error(f'Something went wrong checking the travel time: {str(e)}')
        errors.append(f'Something went wrong checking the travel time: {str(e)}')
    
    return errors

def plot_schedule_from_excel(uploaded_file):
    """Plot een Gantt-grafiek voor busplanning op basis van een Excel-bestand."""

    uploaded_file['starttijd'] = pd.to_datetime(uploaded_file['starttijd'])
    uploaded_file['eindtijd'] = pd.to_datetime(uploaded_file['eindtijd'])

    uploaded_file['duration'] = (uploaded_file['eindtijd'] - uploaded_file['starttijd']).dt.total_seconds() / 3600

    min_duration = 0.05  
    uploaded_file['duration'] = uploaded_file['duration'].apply(lambda x: max(x, min_duration))

    color_map = {
        '400.0': 'blue',
        '401.0': 'yellow',
        'materiaal rit': 'green',
        'idle': 'red',
        'opladen': 'orange'
    }

  
    uploaded_file['buslijn'] = uploaded_file['buslijn'].astype(str)

    def determine_color(row):
        if pd.notna(row['buslijn']) and row['buslijn'] in color_map:
            return color_map[row['buslijn']]  
        elif row['activiteit'] in color_map:
            return color_map[row['activiteit']]  
        else:
            return 'gray' 

    uploaded_file['color'] = uploaded_file.apply(determine_color, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    omloopnummers = uploaded_file['omloop nummer'].unique()
    omloop_indices = {omloop: i for i, omloop in enumerate(omloopnummers)}

    for omloop in omloopnummers:
        trips = uploaded_file[uploaded_file['omloop nummer'] == omloop]

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


# Define Pages
def bus_checker_page():
    st.title('Bus Planning Checker')
    
    # Bestand uploaden
    uploaded_file = st.file_uploader('Upload Your Bus Planning Here (xlsx)', type=['xlsx'], key='file1') # dit is de data die erin komt
    given_data = st.file_uploader('Upload Your Time Table Here (xlsx)', type=['xlsx'], key='file2') # dit is de data die erin komt
    
    if uploaded_file is not None:
        if given_data is not None:
            try:
                # Probeer het Excel-bestand te lezen
                bus_planning = pd.read_excel(uploaded_file)
                time_table = pd.read_excel(given_data, sheet_name='Dienstregeling')
                distance_matrix = pd.read_excel(given_data, sheet_name='Afstandsmatrix')
                st.write('Your Bus Planning:')
                st.dataframe(bus_planning)
               
                plot_schedule_from_excel(bus_planning)

                # Valideer de data
                validation_errors = validate_schedule(bus_planning,time_table,distance_matrix)
                # Toon elke fout in de lijst als een foutmelding

                if validation_errors:
                    st.error('There were mistakes found in your bus planning:')
                    for error in validation_errors:
                        st.error(error)
                else:
                    st.success('Your bus planning is valid!')
            except Exception as e:
                st.error(f'Something went wrong while trying to read the files: {str(e)}')

                

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