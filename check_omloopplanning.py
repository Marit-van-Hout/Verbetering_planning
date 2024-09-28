import pandas as pd
from datetime import datetime, timedelta
from wiskundig_model import charging

# je gooit dus het omloopschema erin en dan kijk je op het toelaatbaar is
# er moet rekening mee gehouden worden dat de omlopen veranderd kunnen worden, dat is nu nog niet het geval
# er moet rekening mee gehouden worden dat in idle er niet opgeladen wordt, maar pas wanneer er van omloop gewisseld wordt
# er moet nog wat geregeld worden met het opladen maar geen idee hoe

#Bij de laatste stop mag de batterij niet onder de 10% zitten, is dit wel het geval dat geeft het een error # Geen energie meer 
#kijken of de oplaadtijd meer dan 15 minuten is 
#Bus moet teleporteren: eindlocatie is ook beginlocatie 
#Kijken of het overeenkomt met planning 
#Zorgen dat elke rit gecoverd is 
#Uiteindelijke eindlocatie en beginlocatie is de garage  
#Reistijd ligt tussen minimale reistijd en maximale reistijd 
#Tussen alles zit een idle

circuit_planning = pd.read_excel('omloopplanning.xlsx')

max_capacity = 300 # maximale capaciteit in kWH
SOH = [85, 95] # State of Health
charging_speed_90 = 450 / 60 # kwh per minuut bij opladen tot 90%
charging_time_10 = 60 / 60 # kwh per minuut bij oladen tot 10%
actual_capacity_85 = max_capacity * 0.85 # (255 kWh)
actual_capacity_95 = max_capacity * 0.95 # (285 kWh)
actual_capacity = [actual_capacity_85, actual_capacity_95]
daytime_limit = [actual_capacity_85*0.9, actual_capacity_95*0.9]
consumption_per_km = [0.7, 2.5] # kWh per km

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