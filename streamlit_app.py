import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
#uploaded_file = pd.read_excel('omloopplanning.xlsx')
#bus_planning_2 = pd.read_excel('omloopplanning.xlsx')  


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

    # Parameters
    max_capacity = 300 # maximum capacity in kWh
    SOH = 0.9 # State of Health
    charging_speed_90 = 450 / 60 # kWh per minute when charging to 90%
    actual_capacity = max_capacity * SOH
    consumption_per_km = (0.7 + 2.5) / 2 # kWh per km
    min_idle_time = 15
    # charging_time_10 = 60 / 60 # kWh per minute when charging from 90% to 100% (wordt niet gebruikt)
    # daytime_limit = actual_capacity * 0.9 (wordt niet gebruikt)
    
    # Data Preparation
    distance_matrix['afstand in km'] = distance_matrix['afstand in meters'] / 1000
    distance_matrix['min reistijd in uur'] = distance_matrix['min reistijd in min'] / 60
    distance_matrix['max reistijd in uur'] = distance_matrix['max reistijd in min'] / 60
    distance_matrix['mean reistijd in uur'] = (distance_matrix['min reistijd in uur'] + distance_matrix['max reistijd in uur']) / 2
    distance_matrix['buslijn'] = distance_matrix['buslijn'].fillna('deadhead trip')
    distance_matrix['max_energy'] = distance_matrix['afstand in km'] * 2.5
    distance_matrix['min_energy'] = distance_matrix['afstand in km'] * 0.7
    distance = distance_matrix['afstand in km']

    start_tijden = []
    eind_tijden = []
    start_times = time_table['vertrektijd']
    time_table['Row_Number'] = time_table.index + 1
    time_table['vertrektijd_dt'] = time_table['vertrektijd'].apply(lambda x: datetime.strptime(x, '%H:%M'))
    time_table['vertrektijd'] = pd.to_datetime(time_table['vertrektijd'], format='%H:%M', errors='coerce')

    # Functies
    def calculate_end_time(row):
        """Adds the mean travel time to the departure time to create a column with end time in dataframe time_table.
    
        Parameters: row (DataFrame row)
        Output: end time in HH:MM
        """
        travel_time = distance_matrix[(distance_matrix['startlocatie'] == row['startlocatie']) & 
                                      (distance_matrix['eindlocatie'] == row['eindlocatie'])]['mean reistijd in uur'].values
        if len(travel_time) > 0:  # Check if travel_time is not empty
            travel_time_in_min = travel_time[0] * 60  # Convert travel time to minutes
            end_time = row['vertrektijd_dt'] + timedelta(minutes=travel_time_in_min)
            return end_time
        else:
            return None

    time_table['eindtijd'] = time_table.apply(calculate_end_time, axis=1)


    def simulate_battery(bus_planning, actual_capacity, global_start_time, global_end_time):
        """Simulate battery usage throughout the day based on the bus planning."""
        battery = actual_capacity * 0.9
        min_battery = actual_capacity * 0.1

        # Iterate over each row in the bus planning
        for i, row in bus_planning.iterrows():
            trip_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')  # Renamed to avoid conflict
            trip_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')  # Renamed to avoid conflict
        
            # Check if the trip is a regular or deadhead trip
            if row['activiteit'] in ['dienst rit', 'materiaal rit']:
                consumption = row['energieverbruik']  # Energy consumption for this trip
                battery -= consumption  # Subtract from battery

                # If the battery falls below the minimum threshold
                if battery < min_battery:
                    st.error(f"Battery of bus {row['omloop nummer']:.0f} too low at {row['starttijd']}.")

            # Check if the bus has enough time to charge
            elif row['activiteit'] == 'opladen':  # Activity is charging
                charging_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
                charging_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
                idle_time = (charging_end_time - charging_start_time).total_seconds() / 60  # Idle time in minutes

                min_idle_time = 10  # Assuming 10 minutes is the minimum idle time required for charging

                if idle_time >= min_idle_time:
                    battery = charging(battery, actual_capacity, charging_start_time, global_start_time, global_end_time)
                else:
                    st.error(f"Charging time too short between {row['starttijd']} and {row['eindtijd']}, only {idle_time} minutes.")

        # Ensure battery remains above 10%
        if battery < min_battery:
            st.error(f"Battery too low after {row['starttijd']}.")
    
        return battery

    
    def start_day(line):
        """
        Deze functie kijkt naar de kolom 'startlocatie' in time_table. 
        Voor elke lijn wordt de tijd van een rit met materiaal (buslijn = NaN) vanuit 'ehvgar' 
        naar de startlocatie bepaald voor beide startlocaties ('ehvapt' en 'ehvbst').
    
        input: de lijn waarvoor de starttijden worden berekend (bijv. 400 of 401).
        output: de berekende starttijden worden toegevoegd aan de lijst start_tijden.
        """
        if line in time_table['buslijn'].values:
            # Verkrijg de rijen van de buslijn
            line_rows = time_table.loc[time_table['buslijn'] == line]

            # Loop over alle rijen van de buslijn
            for index, row in line_rows.iterrows():
                start_locatie = row['startlocatie']

                if start_locatie in ['ehvapt', 'ehvbst']:
                    # Maak een mask om de juiste rij uit de distance_matrix te filteren
                    mask = (
                        (distance_matrix['eindlocatie'] == start_locatie) &
                        (distance_matrix['startlocatie'] == 'ehvgar') &
                        (distance_matrix['buslijn'].isna())
                    )

                    # Controleer of er resultaten zijn
                    if not distance_matrix[mask].empty:
                        # Verkrijg de reistijd
                        reistijd = distance_matrix.loc[mask, 'min reistijd in min'].iloc[0]
                        # Converteer reistijd naar een Timedelta in minuten
                        reistijd_delta = pd.Timedelta(minutes=reistijd)

                        # Bereken de starttijd van de dag
                        start_day = row['vertrektijd'] - reistijd_delta

                        # Voeg de starttijd toe aan de lijst
                        start_tijden.append((line, start_locatie, start_day))  # Voeg lijn en locatie toe
                    else:
                        st.error(f'Row {index + 2}: No matching rides from ehvgar to {start_locatie} for line {line} at the start of the day') # index +2 omdat pandas bij 0 begin te tellen zonder kolomnamen, excel begint bij 1 en telt kolomnamen als een eigen rij
                else:
                    st.error(f'Row {index + 2}: Starting location not recognized: {start_locatie}')
        else:
            st.error(f'Row {index + 2}: No rides found for bus line {line}')



    def end_day(line):
        """
        Deze functie berekent de eindtijden van de dag voor een bepaalde buslijn, vanuit zowel 'ehvapt' als 'ehvbst'.
        Voor beide locaties wordt de laatste vertrektijd genomen en gecombineerd met de reistijd naar 'ehvgar' uit de distance_matrix.

        input: de lijn waarvoor de eindtijden worden berekend (bijv. 400 of 401)
        output: de berekende eindtijden worden toegevoegd aan de lijst eind_tijden.
        """
        if line in time_table['buslijn'].values:
            # Controleer beide locaties: 'ehvapt' en 'ehvbst'
            for locatie in ['ehvapt', 'ehvbst']:
                # Filter de laatste rit vanuit de huidige locatie
                mask = (time_table['buslijn'] == line) & (time_table['eindlocatie'] == locatie)  # dus 'ehvapt' of 'ehvbst'
            
                if not time_table[mask].empty:
                    laatste_rit = time_table[mask].iloc[-1]  # kijk naar de laatste rij
                    eind_vertrektijd = laatste_rit['vertrektijd']  # laatste vertrektijd
                
                    # Zorg ervoor dat eind_vertrektijd een datetime object is
                    eind_vertrektijd = pd.to_datetime(eind_vertrektijd)

                    # Zoek de reistijd naar 'ehvgar' in de distance_matrix
                    reistijd_mask = (
                        (distance_matrix['startlocatie'] == locatie) &  # zorg dat de startlocatie 'ehvapt' of 'ehvbst'is.
                        (distance_matrix['eindlocatie'] == 'ehvgar') &  # zorg dat de eindlocatie 'ehvgar' is
                        (distance_matrix['buslijn'].isna())  # zorg dat je kijkt naar de materiaal ritten
                    )

                    if not distance_matrix[reistijd_mask].empty:
                        # Verkrijg de reistijd en bereken de eindtijd
                        reistijd = distance_matrix.loc[reistijd_mask, 'min reistijd in min'].iloc[0]
                        reistijd_delta = pd.Timedelta(minutes=reistijd)
                        eind_dag_tijd = eind_vertrektijd + reistijd_delta

                        # Voeg de eindtijd toe aan de lijst met de locatie vermeld
                        eind_tijden.append((line, locatie, eind_dag_tijd))
                    else:
                        # Get the index of the last row for more specific error messaging
                        last_index = laatste_rit.name  # the name property gives the index
                        st.error(f"Row {last_index + 2}: No matching rides from {locatie} to 'ehvgar' for line {line} at the end of the day")
                else:
                    # Get any row index for better error messaging
                    first_index = time_table[mask].index.min() if not time_table[mask].empty else None
                    if first_index is not None:
                        st.error(f"Row {first_index + 2}: No rides found for bus line {line} with end location {locatie} at the end of the day")
                    else:
                        st.error(f"Row {first_index + 2}: No rides found for bus line {line} with end location {locatie} at the end of the day")
        else:
            st.error(f"No rides found for bus line {line} at the end of the day")


    # Fleurs creatie. 
    def current_time(time_table):
        """
        Deze functie houdt bij waar we zijn in het schema en laat de tijd zien.
        Input: 
            time_table: DataFrame met de dienstregeling, inclusief vertrektijden en eindtijden.
        Output: 
            current_time: De huidige tijd in het schema.
        """
        # Huidige tijd instellen op het moment dat de functie wordt aangeroepen
        now = datetime.now().time()

        # Zoek de eerste vertrektijd die groter is dan of gelijk aan de huidige tijd
        for index, row in time_table.iterrows():
            departure_time = row['vertrektijd_dt'].time() 
            if departure_time >= now:
                return departure_time  # Geef de eerstvolgende vertrektijd terug

        # Als er geen vertrektijd meer is, geef dan de laatste vertrektijd terug
        return time_table['vertrektijd_dt'].iloc[-1].time()


    def charging(battery, actual_capacity, current_time, start_times, end_times):
        """Simulate battery charging."""
    
        min_battery = 0.10 * actual_capacity
        max_battery_day = 0.90 * actual_capacity
        max_battery_night = actual_capacity
        charging_per_min = charging_speed_90  # Assuming this is defined somewhere else in your code
    
        # Get valid start and end times
        start_time = None
        end_time = None
    
        for line, locatie, tijd in start_times:
            if current_time >= tijd.time():
                start_time = tijd
    
        for line, locatie, tijd in end_times:
            if current_time >= tijd.time():
                end_time = tijd
    
        if start_time is None:
            st.error(f"No valid start time found for current time: {current_time}")
        if end_time is None:
            st.error(f"No valid end time found for current time: {current_time}")
    
        # Determine maximum battery capacity based on time of day
        if current_time < start_time.time() or current_time > end_time.time():
            max_battery = max_battery_night
        else:
            max_battery = max_battery_day
    
        # Calculate new battery capacity
        charged_energy = min_idle_time * charging_per_min
        new_battery = battery + charged_energy if battery <= min_battery else battery
        return min(new_battery, max_battery)


    def battery_consumption(distance, current_time, start_times, end_times):
        """Calculate battery consumption based on distance and current time."""
    
        # Assume max_capacity and consumption_per_km are defined globally
        battery_capacity = max_capacity * 0.9
        consumption = distance * np.mean(consumption_per_km)
        remaining_battery = battery_capacity - consumption
    
        # Get valid start and end times
        start_time = None
        end_time = None
    
        for line, locatie, tijd in start_times:
            if current_time >= tijd.time():
                start_time = tijd
    
        for line, locatie, tijd in end_times:
            if current_time >= tijd.time():
                end_time = tijd
    
        if start_time is None:
            st.error(f"No valid start time found for current time {current_time}")
        if end_time is None:
            st.error(f"No valid end time found for current time {current_time}")
    
        # Call the charging function to update the remaining battery
        return charging(remaining_battery, battery_capacity, current_time, start_times, end_times)


    def check_route_continuity(bus_planning): # de bus kan niet vliegen
        """
        Check if the endpoint of route n matches the start point of route n+1.
        Parameters:
         - bus_planning: DataFrame with route data.
        Output: Print messages if there are inconsistencies.
        """
        for i in range(len(bus_planning) - 1): 
            current_end_location = bus_planning.iloc[i]['eindlocatie']
            next_start_location = bus_planning.iloc[i + 1]['startlocatie']
            if current_end_location != next_start_location:
                st.error(f'Route continuity issue between {bus_planning.iloc[i]['omloop nummer']:.0f} ending at {current_end_location} and next route starting at {next_start_location}.')
                return False
           
        return True


    def driven_rides(bus_planning):
        """ displays which rides are droven
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
        if not difference_time_table_to_bus_planning.empty:
            st.error('Rows only contained in time table:\n', difference_time_table_to_bus_planning)

        if difference_bus_planning_to_time_table.empty and difference_time_table_to_bus_planning.empty:
            return 'Bus planning is equal to time table'


    def plot_schedule_from_excel(bus_planning):
        """Plot een Gantt-grafiek voor busplanning op basis van een Excel-bestand."""
    
        # Zorg ervoor dat de juiste datatypes zijn ingesteld
        date_format = '%Y-%m-%d %H:%M:%S'  # Specify the date format of your columns

        bus_planning['starttijd'] = pd.to_datetime(bus_planning['starttijd'], format=date_format)
        bus_planning['eindtijd'] = pd.to_datetime(bus_planning['eindtijd'], format=date_format)

    
        # Bereken de duur in uren
        bus_planning['duration'] = (bus_planning['eindtijd'] - bus_planning['starttijd']).dt.total_seconds() / 3600

        # Kleurmap voor verschillende buslijnen
        color_map = {
            '400.0': 'blue',
            '401.0': 'yellow'
        }

        # Zet de buslijnwaarden om naar strings
        bus_planning['buslijn'] = bus_planning['buslijn'].astype(str)

        # Voeg een nieuwe kolom toe met de kleur op basis van de buslijn
        bus_planning['color'] = bus_planning['buslijn'].map(color_map).fillna('gray')

        # Maak een figuur voor het plotten
        fig, ax = plt.subplots(figsize=(12, 6))

        # Omloopnummers op de Y-as
        omloopnummers = bus_planning['omloop nummer'].unique()
        omloop_indices = {omloop: i for i, omloop in enumerate(omloopnummers)}

        # Loop door de unieke omloopnummers
        for omloop in omloopnummers:
            trips = bus_planning[bus_planning['omloop nummer'] == omloop]
        
            # Controleer of er ritten zijn
            if trips.empty:
                # Voeg een zwart blok toe als er geen ritten zijn
                ax.barh(omloop_indices[omloop], 1, left=0, color='black', edgecolor='black')
                continue
        
            # Plot elke trip voor de huidige omloop
            for _, trip in trips.iterrows():
                starttime = trip['starttijd']
                duration = trip['duration']
                color = trip['color']  # Haal de kleur direct uit de DataFrame

                # Plot de busrit als een horizontale balk
                ax.barh(omloop_indices[omloop], duration, left=starttime.hour + starttime.minute / 60,
                        color=color, edgecolor='black', label=trip['buslijn'] if trip['buslijn'] not in ax.get_legend_handles_labels()[1] else '')
    
        # Zet de Y-ticks en labels voor de omloopnummers
        ax.set_yticks(list(omloop_indices.values()))
        ax.set_yticklabels(list(omloop_indices.keys()))

        # Set axis labels and title
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Bus Number')
        ax.set_title('Gantt Chart for Bus Scheduling')

        # Voeg een legenda toe (voorkom dubbele labels)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), title='Bus Lines')

        plt.show()


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


    def remove_startingtime_endtime_equal(bus_planning): 
        """ If the starting time and end time are equal, the row is removed.
        Parameters: 
            bus_planning: DataFrame
                Whole bus planning.
        Output: DataFrame
            Clean DataFrame.
        """
        clean_bus_planning = bus_planning[bus_planning['starttijd'] != bus_planning['eindtijd']]
        return clean_bus_planning

    # De validatiefuncties aanroepen
    try:
        time_table['end_time'] = time_table.apply(calculate_end_time, axis=1)
    except Exception as e:
        st.error(f'Something went wrong calculating the end time: {str(e)}')
    
    try: 
        # Roep de functie aan voor beide buslijnen
        start_day(400)
        start_day(401)
    except Exception as e:
        st.error(f'Something went wrong calculating the start of day: {str(e)}')

    try: 
        # Roep de functie aan voor beide buslijnen
        end_day(400)
        end_day(401)
    except Exception as e:
        st.error(f'Something went wrong calculating the end of day: {str(e)}')
    
    try:
        current_time_val = current_time(time_table)
    except Exception as e:
        st.error(f'Something went wrong determining the current time: {str(e)}')
    
    try:
        battery = simulate_battery(bus_planning, actual_capacity, time_table['vertrektijd'], time_table['eindtijd'])
    except Exception as e:
        st.error(f"Something went wrong while simulating the battery: {str(e)}")
    
    try:
        charging(battery, actual_capacity, current_time_val, start_times, time_table['eindtijd'])
    except Exception as e:
        st.error(f"Something went wrong charging the battery: {str(e)}")
    
    try:
        battery_consumption(distance, current_time_val, start_times, time_table['eindtijd'])
    except Exception as e:
        st.error(f"Something went wrong calculating battery consumption: {str(e)}")
    
    try:
        check_route_continuity(bus_planning)
    except Exception as e:
        st.error(f'Something went wrong checking route continuity: {str(e)}')
    
    try:
        driven_rides(bus_planning)
    except Exception as e:
        st.error(f'Something went wrong checking driven rides: {str(e)}')
    
    try:
        every_ride_covered(bus_planning, time_table)
    except Exception as e:
        st.error(f'Something went wrong checking if each ride is covered: {str(e)}')
    
    try:
        plot_schedule_from_excel(bus_planning)
    except Exception as e:
        st.error(f'Something went wrong plotting the bus planning: {str(e)}')
    
    try:
        check_travel_time(bus_planning, distance_matrix)
    except Exception as e:
        st.error(f'Something went wrong checking the travel time: {str(e)}')
    
    try:
        remove_startingtime_endtime_equal(bus_planning)
    except Exception as e:
        st.error(f'Something went wrong deleting equal start and end times: {str(e)}')
    
# Display the logo
st.image('tra_logo_rgb_HR.png', width=200)

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
    st.title('How It Works')
    st.write('This section explains how it works.')

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