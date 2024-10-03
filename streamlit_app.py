import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from io import StringIO
from datetime import datetime, timedelta

# we gaan een app maken die kijkt naar een omloop schema en kijkt of dit omloopschema voldoet aan alle constraints. 
# Zo niet moet er een error komen die zecht: Sorry, maar je stomme bestand werkt niet. Dit is waarom: .... Wat ben je een sukkel

max_capacity = 300 # maximale capaciteit in kWH
SOH = [85, 95] # State of Health
charging_speed_90 = 450 / 60 # kwh per minuut bij opladen tot 90%
charging_time_10 = 60 / 60 # kwh per minuut bij oladen tot 10%
actual_capacity_85 = max_capacity * 0.85 # (255 kWh)
actual_capacity_95 = max_capacity * 0.95 # (285 kWh)
actual_capacity = [actual_capacity_85, actual_capacity_95]
daytime_limit = [actual_capacity_85*0.9, actual_capacity_95*0.9]
consumption_per_km = [0.7, 2.5] # kWh per km
min_idle_time = 15

# Functie om batterijstatus te berekenen
def simulate_battery(circuit_planning, actual_capacity, start_time, end_time):
    """
    Simuleer de batterijstatus gedurende de omloopplanning.
    Parameters:
        - circuit_planning: DataFrame met de omloopplanning.
        - actual_capacity: Batterijcapaciteit van de bus.
        - start_time: Eerste vertrektijd van de dienst.
        - end_time: Laatste eindtijd van de dienst.
    Output: Batterijpercentage na de simulatie.
    """
    battery = actual_capacity * 0.9  # Begin met 90% batterij
    min_battery = actual_capacity * 0.1  # Minimum batterijpercentage
    max_battery_day = actual_capacity * 0.9  # Maximaal 90% overdag
    max_battery_night = actual_capacity  # Maximaal 100% 's nachts
    min_charging_time = 15  # Minimaal 15 minuten opladen
    
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

    for i, row in circuit_planning.iterrows():
        # Converteer start en eindtijden naar datetime
        start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
        end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')

        # Controleer of de rit een dienst of materiaalrit is
        if row['activiteit'] in ['dienst rit', 'materiaal rit']:
            consumption = row['energieverbruik']
            battery -= consumption

         # Controleer of de batterijstatus onder 10% is gekomen
            if battery < min_battery:
                print(f"Warning: Battery too low after route {row['buslijn']} at {row['starttijd']} from {row['startlocatie']} to {row['eindlocatie']}.")
                return battery  # Stop simulation if battery is too low

        # Controleer of de bus idle is en genoeg tijd heeft om op te laden
        if row['activiteit'] == 'opladen':
            idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            idle_time = (idle_end_time - idle_start_time).total_seconds() / 60  # Idle tijd in minuten

            # Controleer of de idle tijd minstens 15 minuten is
            if idle_time >= min_charging_time:
                battery = charging(battery, actual_capacity, idle_start_time, start_time, end_time)
            else:
                print(f"Warning: Charging time too short at Ehvgar from {row['starttijd']} to {row['eindtijd']}, only {idle_time} minutes.")
        
        # Check batterijstatus na elke stap
        if battery < min_battery:
            print(f"Warning: Battery too low after {row['starttijd']}.")
            break

    return battery


# Functie om routecontinuïteit te controleren
def check_route_continuity(circuit_planning):
    """
    Controleer of het eindpunt van route n overeenkomt met het startpunt van route n+1.
    Parameters:
        - circuit_planning: DataFrame met routegegevens.
    Output: Print meldingen als er inconsistenties zijn.
    """
    for i in range(len(circuit_planning) - 1):
        current_end_location = circuit_planning.iloc[i]['eindlocatie']
        next_start_location = circuit_planning.iloc[i+1]['startlocatie']
        if current_end_location != next_start_location:
            print(f"Warning: Route continuity issue between {circuit_planning.iloc[i]['buslijn']} ending at {current_end_location} and next route starting at {next_start_location}.")
            return False
    return True


# Voorbeeld simulatie van batterijverbruik en routecontinuïteit
actual_capacity = 285  # Capaciteit van de bus in kWh
starting_time = datetime.strptime('06:00', '%H:%M')
end_time = datetime.strptime('00:00', '%H:%M')

# Controleer de route continuïteit
if check_route_continuity(circuit_planning):
    # Voer de simulatie uit
    final_battery = simulate_battery(circuit_planning, actual_capacity, starting_time, end_time)
    print(f"Battery status at the end of the day: {final_battery:.2f} kWh")
else:
    print("The circuit planning is not usable due to routing continuity problems.")

# Voorbeeld data
actual_capacity = 285  # Capaciteit van de bus
starting_time = datetime.strptime('06:00', '%H:%M')
end_time = datetime.strptime('00:00', '%H:%M')

# Voer de simulatie uit
simulate_battery(circuit_planning, actual_capacity, starting_time, end_time)
check_route_continuity(circuit_planning)

st.title("🚌 Circulation Planning Checker")
st.write(
    "Instantly validate your circulation planning for compliance!"
) 

# Bestand uploaden (CSV of Excel)
uploaded_file = st.file_uploader("Drag and drop your circulation planning (CSV or Excel file)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Lees het geüploade bestand (CSV of Excel)
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        st.write("**Your circulation planning:**")
        st.dataframe(data)
        
        # Validatie functie (voorbeeld)
        def validate_schema(df):
            errors = []
            # Controleer op missende waarden
            if data.isnull().values.any():
                errors.append("Er zijn missende waarden in het omloopschema.")
                
            # Controleer of de batterijstatus na de laatste rit boven 10% is
            final_battery = simulate_battery(data, actual_capacity, starting_time, end_time)
            if final_battery < (actual_capacity * 0.1):
                errors.append("De batterij is onder de 10% aan het einde van de rit.")
            
            # Controleer oplaadtijd
            for i, row in data.iterrows():
                if row['activiteit'] == 'opladen':
                    idle_start_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
                    idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
                    idle_time = (idle_end_time - idle_start_time).total_seconds() / 60  # Idle tijd in minuten
                    if idle_time < 15:
                        errors.append(f"Oplaadtijd is te kort in rit {row['buslijn']} van {row['startlocatie']} naar {row['eindlocatie']}.")

            # Controleer teleportatie
            if data.iloc[-1]['eindlocatie'] != data.iloc[0]['startlocatie']:
                errors.append("De eindlocatie van de laatste rit komt niet overeen met de beginlocatie van de eerste rit.")
            
            # Controleer reistijden, coverage, en idle tijd
            for i in range(len(data) - 1):
                current_end_location = data.iloc[i]['eindlocatie']
                next_start_location = data.iloc[i + 1]['startlocatie']
                # Check of eindlocatie van de huidige rit overeenkomt met de startlocatie van de volgende rit
                if current_end_location != next_start_location:
                    errors.append(f"Route continuïteit probleem tussen rit {data.iloc[i]['buslijn']} en {data.iloc[i + 1]['buslijn']}.")
            
            return errors
        
        # Voer validatie uit
        validation_errors = validate_schema(data)
        
        if validation_errors:
            st.error("File processed successfully. A few mistakes were found in the circulation planning:")
            for error in validation_errors:
                st.error(error)
        else:
            st.success("File processed successfully. No errors found in the bus planning.")
                   
            # Optionele visualisatie
            st.write("**Visualisation of the circulation planning:**")
            fig, ax = plt.subplots()
            ax.scatter(data['speed'], data['energy'])
            ax.set_xlabel('Speed (km/h)')
            ax.set_ylabel('Energy consumption (kWh)')
            ax.set_title('Speed vs energy consumption')
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"There was an error processing the file: {str(e)}")