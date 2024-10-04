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
schedule = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Dienstregeling")

# Parameters
max_capacity = 300 # maximale capaciteit in kWH
SOH = [85, 95] # State of Health
charging_speed_90 = 450 / 60 # kwh per minuut bij opladen tot 90%
charging_time_10 = 60 / 60 # kwh per minuut bij opladen van 90% tot 100%
actual_capacity_90 = max_capacity * 0.9
actual_capacity = actual_capacity_90 
daytime_limit = actual_capacity_90 *0.9
consumption_per_km = (0.7+2.5)/2 # kWh per km
min_idle_time = 15

<<<<<<< HEAD
errors = []

=======
>>>>>>> b9d9c38f0ed100730c9e33743dc4374d4d625726
# Data Preparation
distance_matrix["afstand in km"] = distance_matrix["afstand in meters"] / 1000
distance_matrix["min reistijd in uur"] = distance_matrix["min reistijd in min"] / 60
distance_matrix["max reistijd in uur"] = distance_matrix["max reistijd in min"] / 60
distance_matrix["buslijn"] = distance_matrix["buslijn"].fillna("materiaalrit")
distance_matrix["max_energy"] = distance_matrix["afstand in km"] * 2.5
distance_matrix["min_energy"] = distance_matrix["afstand in km"] * 0.7

schedule['vertrektijd_dt'] = schedule['vertrektijd'].apply(lambda x: datetime.strptime(x, '%H:%M'))

def calculate_end_time(row):
    """ telt de maximale reistijd op bij de vertrektijd, zodat er een kolom komt met eindtijd
    Parameters: row
    Output: eindtijd in HH:MM
    """
    travel_time = distance_matrix[(distance_matrix['startlocatie'] == row['startlocatie']) & 
                              (distance_matrix['eindlocatie'] == row['eindlocatie'])]['min reistijd in uur'].values
    if len(travel_time) > 0:  # Use len(travel_time) > 0 to check if travel_time is not empty
        travel_time_in_min = travel_time[0] * 60  # Converteer reistijd naar minuten
        end_time = row['vertrektijd_dt'] + timedelta(minutes=travel_time_in_min)
        return end_time
    else:
        return None


schedule['eindtijd'] = schedule.apply(calculate_end_time, axis=1)

# Battery charging simulation
def charging(battery, actual_capacity, current_time, start_time, end_time):
    """Charge the battery based on the current time and bus schedule."""
    min_battery = 0.10 * actual_capacity
    max_battery_day = 0.90 * actual_capacity
    max_battery_night = actual_capacity
    charging_per_min = charging_speed_90

    if current_time < start_time or current_time > end_time:
        max_battery = max_battery_night
    else:
        max_battery = max_battery_day

    charged_energy = min_idle_time * charging_per_min
    new_battery = battery + charged_energy if battery <= min_battery else battery
    return min(new_battery, max_battery)

def simulate_battery(uploaded_file, actual_capacity, start_time, end_time):
    """Simulate battery usage throughout the day based on the circulation plan."""
    battery = actual_capacity * 0.9
    min_battery = actual_capacity * 0.1

    # Converteer start en eindtijden naar datetime
    for i, row in uploaded_file.iterrows():
        start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
        end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
        
         # Controleer of de rit een dienst of materiaalrit is
        if row['activiteit'] in ['dienst rit', 'materiaal rit']:
            consumption = row['energieverbruik']
            battery -= consumption
            if battery < min_battery:
                errors.append(f"Warning: Battery of bus {row['omloop nummer']:.0f} too low at {row['starttijd']}.")
        
        # Controleer of de bus genoeg tijd heeft om op te laden
        elif row['activiteit'] == 'opladen':
            idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            idle_time = (idle_end_time - idle_start_time).total_seconds() / 60
            if idle_time >= min_idle_time:
                battery = charging(battery, actual_capacity, idle_start_time, start_time, end_time)
            else:
                errors.append(f"Warning: Charging time too short between {row['starttijd']} and {row['eindtijd']}, only {idle_time} minutes.")

        # Controleer of de bus altijd meer dan 10% volle batterij heeft
        if battery < min_battery:
            errors.append(f"Warning: Battery too low after {row['starttijd']}.")
    
    return battery

# Functie om routecontinuïteit te controleren
def check_route_continuity(circulation_planning):
    """
    Controleer of het eindpunt van route n overeenkomt met het startpunt van route n+1.
    Parameters:
        - circulation_planning: DataFrame met routegegevens.
    Output: Print meldingen als er inconsistenties zijn.
    """
    for i in range(len(circulation_planning) - 1):
        current_end_location = circulation_planning.iloc[i]['eindlocatie']
        next_start_location = circulation_planning.iloc[i+1]['startlocatie']
        if current_end_location != next_start_location:
<<<<<<< HEAD
            errors.append(f"Warning: Route continuity issue between {circuit_planning.iloc[i]['omloop nummer']:.0f} ending at {current_end_location} and next route starting at {next_start_location}.")
=======
            print(f"Warning: Route continuity issue between {circulation_planning.iloc[i]['omloop nummer']:.0f} ending at {current_end_location} and next route starting at {next_start_location}.")
>>>>>>> b9d9c38f0ed100730c9e33743dc4374d4d625726
            return False
           
    return True

def battery_consumption(distance, current_time, start_time, end_time):
    """Calculate battery consumption based on distance and time."""
    battery_capacity = max_capacity * 0.9
    consumption = distance * np.mean(consumption_per_km)
    remaining_battery = battery_capacity - consumption
    
    return charging(remaining_battery, battery_capacity, current_time, start_time, end_time)

<<<<<<< HEAD
# Run checks and simulation
if check_route_continuity(circuit_planning):
    starting_time = datetime.strptime('06:00', '%H:%M')
    ending_time = datetime.strptime('00:00', '%H:%M')
    final_battery = simulate_battery(circuit_planning, actual_capacity, starting_time, ending_time)
    errors.append(f"Final battery status: {final_battery:.0f} kWh")
else:
    errors.append("Circuit planning failed continuity checks.")
    

=======
>>>>>>> b9d9c38f0ed100730c9e33743dc4374d4d625726
# Yvonnes code
def driven_rides(circulation_planning):
    """ displays which rides are droven
    Parameters
        omloopplanning: DataFrame
        The full circulation planning data.
    output
        DataFrame
        A cleaned circulation planning DataFrame containing only the relevant columns 
        and rows where a bus line is present.
    """
    clean_circulation_planning = circulation_planning[['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']]
    clean_circulation_planning = clean_circulation_planning.dropna(subset=['buslijn'])
    return clean_circulation_planning

def normalize_time_format(df, time_column):
    """Convert time to a uniform format, ignoring seconds
    Parameters: 
        - df : DataFrame
            The DataFrame containing time data
        - time column: str
            column with time as a string
    output: 
        DataFrame
        DataFrame with time in standarized form (%H:%M)
    """
    df[time_column] = pd.to_datetime(df[time_column]).dt.strftime('%H:%M')
    return df

def every_ride_covered(circulation_planning, schedule):
    """Checks if every ride in the timetable is covered in circulation planning.
    
    Parameters: 
        circulation_planning : DataFrame
            The DataFrame representing the rides that are actually driven.
        dienstregeling : DataFrame
            The DataFrame representing the rides that are supposed to be driven.
    
    Returns:
        DataFrame or str
            If there are differences, returns a DataFrame with the differences.
            If all rides are covered, returns a success message.
    """
    schedule = schedule.rename(columns={'vertrektijd': 'starttijd'})
    
    circulation_planning_sorted = circulation_planning.sort_values(by=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']).reset_index(drop=True)
    schedule_sorted = schedule.sort_values(by=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn']).reset_index(drop=True)
    
    difference_circulation_to_schedule = circulation_planning_sorted.merge(
        schedule_sorted, on=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn'], how='outer', indicator=True
    ).query('_merge == "left_only"')

    difference_schedule_to_circulation = circulation_planning_sorted.merge(
        schedule_sorted, on=['startlocatie', 'starttijd', 'eindlocatie', 'buslijn'], how='outer', indicator=True
    ).query('_merge == "right_only"')

    if not difference_circulation_to_schedule.empty:
        print("Rows only contained in circulation planning:\n", difference_circulation_to_schedule)
    if not difference_schedule_to_circulation.empty:
        print("Rows only contained in schedule:\n", difference_schedule_to_circulation)

    if difference_circulation_to_schedule.empty and difference_schedule_to_circulation.empty:
        return "Circulation planning is equal to timetable"
       
uploaded_file = driven_rides(uploaded_file)
uploaded_file = normalize_time_format(uploaded_file, "starttijd")

result = every_ride_covered(uploaded_file, schedule)

print(result)

def check_travel_time(circulation_planning, distance_matrix):
    """
    Controleert of het tijdsverschil tussen de starttijd en eindtijd in circulation_planning
    binnen de minimale en maximale reistijd in distance_matrix ligt, terwijl startlocatie,
    eindlocatie en buslijn gelijk zijn.
    
    Parameters:
    circulation_planning : DataFrame
        Een DataFrame met kolommen 'starttijd', 'eindtijd', 'startlocatie', 'eindlocatie', 'buslijn'.
    distance_matrix : DataFrame
        Een DataFrame met kolommen 'startlocatie', 'eindlocatie', 'minimale_reistijd', 'maximale_reistijd', 'buslijn'.
    
    Returns:
    Meldingen voor rijen die niet binnen de reistijd vallen.
    """
    # Zorg ervoor dat starttijd en eindtijd datetime zijn
    circulation_planning['starttijd'] = pd.to_datetime(circulation_planning['starttijd'], format='%H:%M:%S', errors='coerce')
    circulation_planning['eindtijd'] = pd.to_datetime(circulation_planning['eindtijd'], format='%H:%M:%S', errors='coerce')

    # Bereken het verschil in minuten
    circulation_planning['verschil_in_minuten'] = (circulation_planning['eindtijd'] - circulation_planning['starttijd']).dt.total_seconds() / 60

    # Merge beide datasets op 'startlocatie', 'eindlocatie' en 'buslijn'
    merged_df = pd.merge(
        circulation_planning,
        distance_matrix,
        on=['startlocatie', 'eindlocatie', 'buslijn'],
        how='inner'  # Alleen gemeenschappelijke rijen behouden
    )

    # Controleer of het verschil binnen de minimale en maximale reistijd ligt
    for index, row in merged_df.iterrows():
        if not (row['min reistijd in min'] <= row['verschil_in_minuten'] <= row['max reistijd in min']):
            errors.append(f"Row {index}: The difference in minutes ({row['verschil_in_minuten']:.0f}) is not between {row['max reistijd in min']} and {row['min reistijd in min']} for bus route {row['buslijn']} from {row['startlocatie']} to {row['eindlocatie']}.")
        
# Voorbeeld aanroepen met omloopplanning2 en afstandsmatrix
circulation_planning_2 = pd.read_excel('omloopplanning.xlsx')  # Zorg ervoor dat deze bestaat

# Functie aanroepen
check_travel_time(circulation_planning_2, distance_matrix)

def remove_startingtime_endtime_equal(circulation_planning): 
    """ If the starting time and end time are equal, than the row is being removed
    Parameters: 
        circulation_planning: DataFrame
            Whole circulation planning
    Output: DataFrame
        Clean DataFrame
    """
    clean_circulation_planning = circulation_planning[circulation_planning['starttijd'] != circulation_planning['eindtijd']]
    return clean_circulation_planning

<<<<<<< HEAD
new_planning = remove_startingtime_endtime_equal(circuit_planning)

st.title("🎈 Oploopschema Validatie App")
st.write(
    "Upload je oploopschema (CSV of Excel) en download het gevalideerde schema."
) 

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
        
        # Validatie functie (voorbeeld)
        def validate_schema(df):
            calculate_end_time(row)
            charging(battery, actual_capacity, current_time, start_time, end_time)
            simulate_battery(circuit_planning, actual_capacity, start_time, end_time)
            check_route_continuity(circuit_planning)
            battery_consumption(distance, current_time, start_time, end_time)
            check_route_continuity(circuit_planning)
            driven_rides(circuit_planning)
            normalize_time_format(df, time_column)
            every_ride_covered(circuit_planning, schedule)
            check_travel_time(circuit_planning, distance_matrix)
            remove_startingtime_endtime_equal(circuit_planning)
            return errors
        
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
=======
new_planning = remove_startingtime_endtime_equal(uploaded_file)
>>>>>>> b9d9c38f0ed100730c9e33743dc4374d4d625726
