import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from io import StringIO
from datetime import datetime, timedelta
from wiskundig_model import charging
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
uploaded_file = pd.read_excel('omloopplanning.xlsx')
distance_matrix = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Afstandsmatrix")
time_table = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Dienstregeling")

# Parameters
max_capacity = 300 # maximum capacity in kWh
SOH = [85, 95] # State of Health
charging_speed_90 = 450 / 60 # kWh per minute when charging to 90%
charging_time_10 = 60 / 60 # kWh per minute when charging from 90% to 100%
actual_capacity_90 = max_capacity * 0.9
actual_capacity = actual_capacity_90 
daytime_limit = actual_capacity_90 * 0.9
consumption_per_km = (0.7 + 2.5) / 2 # kWh per km
min_idle_time = 15


errors = []


# Data Preparation
distance_matrix["afstand in km"] = distance_matrix["afstand in meters"] / 1000
distance_matrix["min reistijd in uur"] = distance_matrix["min reistijd in min"] / 60
distance_matrix["max reistijd in uur"] = distance_matrix["max reistijd in min"] / 60
distance_matrix["mean reistijd in uur"] = (distance_matrix["min reistijd in uur"] + distance_matrix["max reistijd in uur"]) / 2
distance_matrix["buslijn"] = distance_matrix["buslijn"].fillna("deadhead trip")
distance_matrix["max_energy"] = distance_matrix["afstand in km"] * 2.5
distance_matrix["min_energy"] = distance_matrix["afstand in km"] * 0.7

time_table['Row_Number'] = time_table.index + 1

time_table['vertrektijd_dt'] = time_table['vertrektijd'].apply(lambda x: datetime.strptime(x, '%H:%M'))

# fleur: ik denk dat dit een fout is geweest maar moet dit niet gebaseerd zijn op de "max reistijd in min" en niet de "min reistijd in min"
# fleur: is het niet handiger als we de mean ervan nemen en die opstellen als de normale "end_time"
# fleurs aangepaste code:
# aanpassen:
def calculate_end_time(row):
    """ Adds the maximum travel time to the departure time to create a column with end time.
    Parameters: row
    Output: end time in HH:MM
    """
    travel_time = distance_matrix[(distance_matrix['startlocatie'] == row['startlocatie']) & 
                                  (distance_matrix['eindlocatie'] == row['eindlocatie'])]["mean reistijd in uur"].values
    if len(travel_time) > 0:  # Check if travel_time is not empty
        travel_time_in_min = travel_time[0] * 60  # Convert travel time to minutes
        end_time = row['vertrektijd_dt'] + timedelta(minutes=travel_time_in_min)
        return end_time
    else:
        return None
    
#def calculate_end_time_1(row):
    #""" Adds the maximum travel time to the departure time to create a column with end time.
    #Parameters: row
    #Output: end time in HH:MM
    #"""
    #travel_time = distance_matrix[(distance_matrix['startlocatie'] == row['startlocatie']) & 
    #                             (distance_matrix['eindlocatie'] == row['eindlocatie'])]['min reistijd in uur'].values
    #if len(travel_time) > 0:  # Check if travel_time is not empty
    #    travel_time_in_min = travel_time[0] * 60  # Convert travel time to minutes
    #    end_time = row['vertrektijd_dt'] + timedelta(minutes=travel_time_in_min)
    #    return end_time
    #else:
    #    return None

time_table['eindtijd'] = time_table.apply(calculate_end_time, axis=1)

# fleur versie 
def simulate_battery(uploaded_file, actual_capacity, start_time, end_time):
    """Simulate battery usage throughout the day based on the bus planning."""
    battery = actual_capacity * 0.9
    min_battery = actual_capacity * 0.1

    # Convert start and end times to datetime
    for i, row in uploaded_file.iterrows():
        start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
        end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
        
        # Check if the trip is a regular or deadhead trip
        if row['activiteit'] in ['dienst rit', 'materiaal rit']: # als activiteit dienst rit of materiaal rit is:
            consumption = row['energieverbruik'] # dan kijken we naar de consuption rij van deze rit
            battery -= consumption # dit gaat min de batterij die we al hebben
            if battery < min_battery: # als de batterij minder dan 10 is 
                errors.append(f"Warning: Battery of bus {row['omloop nummer']:.0f} too low at {row['starttijd']}.")
        
        # Check if the bus has enough time to charge # dit klopt hememaal: Fleur
        elif row['activiteit'] == 'opladen': # als de activiteit opladen is
            idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            idle_time = (idle_end_time - idle_start_time).total_seconds() / 60
            if idle_time >= min_idle_time:
                battery = charging(battery, actual_capacity, idle_start_time, start_time, end_time)
            else:
                errors.append(f"Warning: Charging time too short between {row['starttijd']} and {row['eindtijd']}, only {idle_time} minutes.")

        # Ensure battery remains above 10%
        if battery < min_battery:
            errors.append(f"Warning: Battery too low after {row['starttijd']}.")
    
    return battery

#def simulate_battery(uploaded_file, actual_capacity, start_time, end_time):
    #"""Simulate battery usage throughout the day based on the bus planning."""
    #battery = actual_capacity * 0.9
    #min_battery = actual_capacity * 0.1

    # Convert start and end times to datetime
    #for i, row in uploaded_file.iterrows():
        #start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
        #end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
        
        # Check if the trip is a regular or deadhead trip
        #if row['activiteit'] in ['regular trip', 'deadhead trip']:
            #consumption = row['energieverbruik']
            #battery -= consumption
            #if battery < min_battery:
                #errors.append(f"Warning: Battery of bus {row['omloop nummer']:.0f} too low at {row['starttijd']}.")
        
        # Check if the bus has enough time to charge
        #elif row['activiteit'] == 'opladen':
            #idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            #idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            #idle_time = (idle_end_time - idle_start_time).total_seconds() / 60
            #if idle_time >= min_idle_time:
                #battery = charging(battery, actual_capacity, idle_start_time, start_time, end_time)
            #else:
                #errors.append(f"Warning: Charging time too short between {row['starttijd']} and {row['eindtijd']}, only {idle_time} minutes.")

        # Ensure battery remains above 10%
        #if battery < min_battery:
            #errors.append(f"Warning: Battery too low after {row['starttijd']}.")
    
    #return battery

# fleurs probeersel van een starttijd van de dag
start_tijden = []

distance_matrix = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Afstandsmatrix")
time_table = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Dienstregeling")

# Probeer de conversie opnieuw
time_table["vertrektijd"] = pd.to_datetime(time_table["vertrektijd"], format='%H:%M', errors='coerce')

# ik krijg hier errors dat er in distance_matrix geen rij gevonden kan worden bij de materiaal ritten (buslijn == NaN)
# waar de startlocatie "ehvgar" is en de eindlocatie "ehvapt" of "ehvbst".
def Start_day(line):
    """
    Deze functie kijkt naar de kolom 'startlocatie' in time_table. 
    Voor elke lijn wordt de tijd van een rit met materiaal (buslijn = NaN) vanuit 'ehvgar' 
    naar de startlocatie bepaald voor beide startlocaties ('ehvapt' en 'ehvbst').
    
    input: de lijn waarvoor de starttijden worden berekend (bijv. 400 of 401).
    output: de berekende starttijden worden toegevoegd aan de lijst start_tijden.
    """
    if line in time_table["buslijn"].values:
        # Verkrijg de startlocaties van de buslijn
        start_locaties = time_table.loc[time_table["buslijn"] == line, "startlocatie"].unique()

        for start_locatie in start_locaties:
            if start_locatie in ["ehvapt", "ehvbst"]:
                # Maak een mask om de juiste rij uit de distance_matrix te filteren
                mask = (
                    (distance_matrix["eindlocatie"] == start_locatie) &
                    (distance_matrix["startlocatie"] == "ehvgar") &
                    (distance_matrix["buslijn"].isna())
                )
                # Controleer of er resultaten zijn
                if not distance_matrix[mask].empty:
                    # Verkrijg de reistijd
                    reistijd = distance_matrix.loc[mask, "min reistijd in min"].iloc[0]
                    # Converteer reistijd naar een Timedelta in minuten
                    reistijd_delta = pd.Timedelta(minutes=reistijd)

                    # Bereken de starttijd van de dag
                    start_day = time_table.loc[time_table["buslijn"] == line, "vertrektijd"].iloc[0] - reistijd_delta
                    
                    # Voeg de starttijd toe aan de lijst
                    start_tijden.append((line, start_locatie, start_day))  # Voeg lijn en locatie toe
                else:
                    errors.append(f"Geen matchende rit gevonden van 'ehvgar' naar {start_locatie} functie Start_day")

            else:
                errors.append("Startlocatie niet herkend:", start_locatie)
    else:
        errors.append("Geen ritten gevonden voor buslijn", line)

# Roep de functie aan voor beide buslijnen
Start_day(400)
Start_day(401)

# Toon de starttijden
print("Starttijden van de dag:", start_tijden)

# fleurs probeersel van een eindtijden van de dag
import pandas as pd

# ik krijg hier errors dat er in distance_matrix geen rij gevonden kan worden bij de materiaal ritten (buslijn == NaN)
# waar de startlocatie "ehvgar" is en de eindlocatie "ehvapt" of "ehvbst".
# Lijst om de eindtijden op te slaan
eind_tijden = []

def eind_dag(line):
    """
    Deze functie berekent de eindtijden van de dag voor een bepaalde buslijn, vanuit zowel 'ehvapt' als 'ehvbst'.
    Voor beide locaties wordt de laatste vertrektijd genomen en gecombineerd met de reistijd naar 'ehvgar' uit de distance_matrix.

    input: de lijn waarvoor de eindtijden worden berekend (bijv. 400 of 401)
    output: de berekende eindtijden worden toegevoegd aan de lijst eind_tijden.
    """
    if line in time_table["buslijn"].values:
        # Controleer beide locaties: 'ehvapt' en 'ehvbst'
        for locatie in ["ehvapt", "ehvbst"]:
            # Filter de laatste rit vanuit de huidige locatie
            mask = (time_table["buslijn"] == line) & (time_table["eindlocatie"] == locatie) # dus "ehvapt" of "ehvbst"
            if not time_table[mask].empty:
                laatste_rit = time_table[mask].iloc[-1] # kijk naar de laatste rij
                eind_vertrektijd = laatste_rit["vertrektijd"] # Nu kijken we naar de allerlaaste dienstrit en daar de vertrektijd van 

                # Zorg ervoor dat eind_vertrektijd een datetime object is
                eind_vertrektijd = pd.to_datetime(eind_vertrektijd)

                # Zoek de reistijd naar 'ehvgar' in de distance_matrix
                reistijd_mask = (
                    (distance_matrix["startlocatie"] == locatie) & # zorg dat de startlocatie "ehvapt" of "ehvbst"is.
                    (distance_matrix["eindlocatie"] == "ehvgar") & # zorg dat de eindlocatie "ehvgar" is
                    (distance_matrix["buslijn"].isna()) # zorg dat je kijkt naar de materiaal ritten
                )
                
                if not distance_matrix[reistijd_mask].empty:
                    # Verkrijg de reistijd en bereken de eindtijd
                    reistijd = distance_matrix.loc[reistijd_mask, "min reistijd in min"].iloc[0]
                    reistijd_delta = pd.Timedelta(minutes=reistijd)
                    eind_dag_tijd = eind_vertrektijd + reistijd_delta
                    
                    # Voeg de eindtijd toe aan de lijst met de locatie vermeld
                    eind_tijden.append((line, locatie, eind_dag_tijd))
                else:
                    errors.append(f"Geen matchende rit gevonden van {locatie} naar 'ehvgar' voor lijn {line} functie eind_dag")
            else:
                errors.append(f"Geen ritten gevonden voor buslijn {line} met eindlocatie {locatie} functie eind_dag")
    else:
        errors.append(f"Geen ritten gevonden voor buslijn {line} functie eind_dag")

# Roep de functie aan voor beide buslijnen
eind_dag(400)
eind_dag(401)

# Toon de eindtijden en eventuele fouten
print("Eindtijden van de dag:", eind_tijden)
if errors:
    print("Fouten:", errors)


# Fleurs creatie. Misshien was het niet gevraagd maar ja
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


# Battery charging simulation
#def charging(battery, actual_capacity, current_time, start_time, end_time): # dit werkt niet als we niet de echte begintijd en eindtijd van de dag hebben.
    #"""Charge the battery based on the current time and time table.""" # ook hebben we geen funtie die de current_time bijhoud.
    #min_battery = 0.10 * actual_capacity
    #max_battery_day = 0.90 * actual_capacity
    #max_battery_night = actual_capacity
    #charging_per_min = charging_speed_90

    #if current_time < start_time or current_time > end_time:
        #max_battery = max_battery_night
    #else:
        #max_battery = max_battery_day

    #charged_energy = min_idle_time * charging_per_min
    #new_battery = battery + charged_energy if battery <= min_battery else battery
    #return min(new_battery, max_battery)

# fleurs creatie: ik zou het niet vertrouwen 
def charging(battery, actual_capacity, current_time, start_times, end_times):
    """
    Simuleert het opladen van de batterij op basis van de huidige tijd en start- en eindtijden van de dienstregeling.
    
    Parameters:
        battery (float): Huidige batterijcapaciteit.
        actual_capacity (float): Totale capaciteit van de batterij.
        current_time (datetime.time): Huidige tijd in het schema.
        start_times (list): Lijst van tuples met (line, locatie, tijd) voor starttijden.
        end_times (list): Lijst van tuples met (line, locatie, tijd) voor eindtijden.
    
    Returns:
        float: Nieuwe batterijcapaciteit na opladen.
        
    Raises:
        ValueError: Als er geen geldige start- of eindtijd wordt gevonden voor de huidige tijd.
    """
    
    min_battery = 0.10 * actual_capacity
    max_battery_day = 0.90 * actual_capacity
    max_battery_night = actual_capacity
    charging_per_min = charging_speed_90
    
    # Zoek de juiste starttijd
    start_time = None
    for line, locatie, tijd in start_times:
        if current_time >= tijd.time():
            start_time = tijd
    
    # Zoek de juiste eindtijd
    end_time = None
    for line, locatie, tijd in end_times:
        if current_time >= tijd.time():
            end_time = tijd
    
    # Controleer of start_time en end_time zijn gevonden
    if start_time is None:
        raise ValueError(f"Geen geldige starttijd gevonden voor de huidige tijd: {current_time}")
    if end_time is None:
        raise ValueError(f"Geen geldige eindtijd gevonden voor de huidige tijd: {current_time}")
    
    # Bepaal maximum batterijlimiet op basis van de tijd
    if current_time < start_time.time() or current_time > end_time.time():
        max_battery = max_battery_night
    else:
        max_battery = max_battery_day

    # Bereken de nieuwe batterijcapaciteit
    charged_energy = min_idle_time * charging_per_min
    new_battery = battery + charged_energy if battery <= min_battery else battery
    return min(new_battery, max_battery)

# fleurs creatie: ik zou deze ook niet vertrouwen
def battery_consumption(distance, current_time, start_times, end_times):
    """
    Bereken het batterijverbruik op basis van de afstand en huidige tijd.
    
    Parameters:
        distance (float): Afstand in kilometers.
        current_time (datetime.time): Huidige tijd in het schema.
        start_times (list): Lijst van tuples met (line, locatie, tijd) voor starttijden.
        end_times (list): Lijst van tuples met (line, locatie, tijd) voor eindtijden.
    
    Returns:
        float: Resterende batterijcapaciteit na verbruik en opladen.
        
    Raises:
        ValueError: Als er geen geldige start- of eindtijd wordt gevonden voor de huidige tijd.
    """
    
    # Bepaal batterijcapaciteit voor de dag
    battery_capacity = max_capacity * 0.9
    
    # Bereken het verbruik op basis van de afstand
    consumption = distance * np.mean(consumption_per_km)
    remaining_battery = battery_capacity - consumption
    
    # Zoek de juiste starttijd
    start_time = None
    for line, locatie, tijd in start_times:
        if current_time >= tijd.time():
            start_time = tijd
    
    # Zoek de juiste eindtijd
    end_time = None
    for line, locatie, tijd in end_times:
        if current_time >= tijd.time():
            end_time = tijd
    
    # Controleer of start_time en end_time zijn gevonden
    if start_time is None:
        raise ValueError(f"Geen geldige starttijd gevonden voor de huidige tijd: {current_time}")
    if end_time is None:
        raise ValueError(f"Geen geldige eindtijd gevonden voor de huidige tijd: {current_time}")
    
    # Roep de charging-functie aan om het resterende batterijpercentage bij te werken
    return charging(remaining_battery, battery_capacity, current_time, start_time, end_time)

def battery_consumption(distance, current_time, start_time, end_time): # hier geld precies hetzelfde.
    """Calculate battery consumption based on distance and time."""
    battery_capacity = max_capacity * 0.9
    consumption = distance * np.mean(consumption_per_km)
    remaining_battery = battery_capacity - consumption
    
    return charging(remaining_battery, battery_capacity, current_time, start_time, end_time)

# Function to check route continuity
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
            errors.append(f"Warning: Route continuity issue between {bus_planning.iloc[i]['omloop nummer']:.0f} ending at {current_end_location} and next route starting at {next_start_location}.")
            return False
           
    return True

# Yvonnes code
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
    clean_bus_planning = clean_bus_planning.dropna(subset=['buslijn'])
    return clean_bus_planning

def normalize_time_format(df, time_column):
    """Convert time to a uniform format, ignoring seconds.
    Parameters: 
        - df : DataFrame
            The DataFrame containing time data.
        - time_column: str
            Column with time as a string.
    Output: 
        DataFrame
        DataFrame with time in standardized form (%H:%M).
    """
    df[time_column] = pd.to_datetime(df[time_column]).dt.strftime('%H:%M')
    return df

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
        errors.append("Rows only contained in bus planning:\n", difference_bus_planning_to_time_table)
    if not difference_time_table_to_bus_planning.empty:
        errors.append("Rows only contained in time table:\n", difference_time_table_to_bus_planning)

    if difference_bus_planning_to_time_table.empty and difference_time_table_to_bus_planning.empty:
        return "Bus planning is equal to time table"


def plot_schedule(scheduled_orders):
    """Plots a Gantt chart of the scheduled orders

    Args:
        scheduled_orders (dict): every order, their starting time, end time, on which machine and set-up time
    """    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    
    # Colors for visualization
    color_map = {
         '400': 'blue',
         '401': 'yellow',
    }
    
    for machine, orders in scheduled_orders.items():
        y_pos += 1  # Voor elke machine
        for order in orders:
            order_color = order['colour']
            processing_time = order['end_time'] - order['start_time'] - order['setup_time']
            setup_time = order['setup_time']
            start_time = order['start_time']
            
            # Controleer of de kleur aanwezig is in de color_map
            if order_color in color_map:
                color = color_map[order_color]
            else:
                color = 'black'  # Default color als de kleur niet bestaat in color_map
            
            # Teken verwerkingstijd
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=color, edgecolor='black')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, f"Order {order['order']}", 
                    ha='center', va='center', color='black', rotation=90)

            # Teken setup tijd
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
    
    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart for Paint Shop Scheduling')
    plt.show()
          
uploaded_file = driven_rides(uploaded_file)
uploaded_file = normalize_time_format(uploaded_file, "starttijd")

result = every_ride_covered(uploaded_file, time_table)

print(result)

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
            errors.append(f"Row {index}: The difference in minutes ({row['verschil_in_minuten']:.0f}) is not between {row['max reistijd in min']} and {row['min reistijd in min']} for bus route {row['buslijn']} from {row['startlocatie']} to {row['eindlocatie']}.")
        
# Example call with bus planning 2 and distance matrix
bus_planning_2 = pd.read_excel('omloopplanning.xlsx')  # Ensure this exists

# Call the function
check_travel_time(bus_planning_2, distance_matrix)

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


#new_planning = remove_startingtime_endtime_equal(bus_planning)

st.title("🎈 Oploopschema Validatie App")
st.write(
    "Upload je oploopschema (CSV of Excel) en download het gevalideerde schema."
) 

def validate_schema(row: dict, time_table: pd.DataFrame, uploaded_file, actual_capacity, start_times, end_times, 
                   distance, bus_planning, df, time_column, scheduled_orders, distance_matrix) -> list[str]:
    """
    Valideert het schema van een busplanning.

    Parameters:
    - row: Een dictionary met de gegevens van de rit.
    - time_table: Een DataFrame met vertrektijden en andere relevante informatie.
    - uploaded_file, actual_capacity, start_times, end_times: Vereiste parameters voor de simulatie en batterijfuncties.
    - distance: Afstand voor batterijverbruik.
    - bus_planning: De planning van de bus.
    - df, time_column: DataFrame en tijdkolom voor normalisatie.
    - scheduled_orders: Geplande orders voor plotting.
    - distance_matrix: Matrix voor reistijdcontrole.

    Returns:
    - Een lijst van foutmeldingen die zijn opgetreden tijdens de validatie.
    """
    errors = []

    def calculate_end_time(row):
        # Implementatie van de functie
        pass

    def current_time(time_table):
        # Implementatie van de functie
        pass

    def simulate_battery(uploaded_file, actual_capacity, departure_time, end_time):
        # Implementatie van de functie
        pass

    def charging(battery, actual_capacity, current_time_val, start_times, end_times):
        # Implementatie van de functie
        pass

    def battery_consumption(distance, current_time_val, start_times, end_times):
        # Implementatie van de functie
        pass

    def check_route_continuity(bus_planning):
        # Implementatie van de functie
        pass

    def driven_rides(bus_planning):
        # Implementatie van de functie
        pass

    def normalize_time_format(df, time_column):
        # Implementatie van de functie
        pass

    def every_ride_covered(bus_planning, time_table):
        # Implementatie van de functie
        pass

    def plot_schedule(scheduled_orders):
        # Implementatie van de functie
        pass

    def check_travel_time(bus_planning, distance_matrix):
        # Implementatie van de functie
        pass

    def remove_startingtime_endtime_equal(bus_planning):
        # Implementatie van de functie
        pass

    # De validatiefuncties aanroepen
    try:
        end_time = calculate_end_time(row)
    except Exception as e:
        errors.append(f"Fout bij berekenen van eindtijd: {str(e)}")
    
    try:
        current_time_val = current_time(time_table)
    except Exception as e:
        errors.append(f"Fout bij ophalen van huidige tijd: {str(e)}")
    
    try:
        battery = simulate_battery(uploaded_file, actual_capacity, time_table['vertrektijd'], end_time)
    except Exception as e:
        errors.append(f"Fout bij simuleren van batterij: {str(e)}")
    
    try:
        charging(battery, actual_capacity, current_time_val, start_times, end_times)
    except Exception as e:
        errors.append(f"Fout bij batterij opladen: {str(e)}")
    
    try:
        battery_consumption(distance, current_time_val, start_times, end_times)
    except Exception as e:
        errors.append(f"Fout bij berekening van batterijverbruik: {str(e)}")
    
    try:
        check_route_continuity(bus_planning)
    except Exception as e:
        errors.append(f"Fout bij controle van routecontinuïteit: {str(e)}")
    
    try:
        driven_rides(bus_planning)
    except Exception as e:
        errors.append(f"Fout bij controle van gereden ritten: {str(e)}")
    
    try:
        normalize_time_format(df, time_column)
    except Exception as e:
        errors.append(f"Fout bij normalisatie van tijdformaat: {str(e)}")
    
    try:
        every_ride_covered(bus_planning, time_table)
    except Exception as e:
        errors.append(f"Fout bij controle of elke rit gedekt is: {str(e)}")
    
    try:
        plot_schedule(scheduled_orders)
    except Exception as e:
        errors.append(f"Fout bij plotten van het schema: {str(e)}")
    
    try:
        check_travel_time(bus_planning, distance_matrix)
    except Exception as e:
        errors.append(f"Fout bij controle van reistijd: {str(e)}")
    
    try:
        remove_startingtime_endtime_equal(bus_planning)
    except Exception as e:
        errors.append(f"Fout bij verwijderen van gelijke start- en eindtijden: {str(e)}")
    
    return errors


# Bestand uploaden (CSV of Excel)
uploaded_file = st.file_uploader("Upload je oploopschema (CSV of Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Lees het geüploade bestand (CSV of Excel)
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        st.write("**Geüpload Oploopschema:**")
        st.dataframe(data)
        
        
        # Voer validatie uit
        validation_errors = validate_schema(data)
        
        if validation_errors:
            st.error("Er zijn fouten gevonden in het oploopschema:")
            for error in validation_errors:
                st.error(error)
        else:
            st.success("Het oploopschema is geldig!")
            
            # Voeg een downloadknop toe voor het gevalideerde schema
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download gevalideerd schema als CSV",
                data=csv,
                file_name='gevalideerd_oploopschema.csv',
                mime='text/csv'
            )
            
            # Optionele visualisatie
            st.write("**Visualisatie van het Oploopschema:**")
            fig, ax = plt.subplots()
            ax.scatter(data['speed'], data['energy'])
            ax.set_xlabel('Snelheid (km/uur)')
            ax.set_ylabel('Energieverbruik (kWh)')
            ax.set_title('Snelheid vs Energieverbruik')
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Er is een fout opgetreden bij het verwerken van het bestand: {str(e)}")

new_planning = remove_startingtime_endtime_equal(uploaded_file)



