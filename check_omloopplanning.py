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
def simulate_battery(circuit_planning, actual_capacity, starting_time, end_time):
    """
    Simuleer de batterijstatus gedurende de omloopplanning.
    Parameters:
        - df: DataFrame met de omloopplanning.
        - DCap: Batterijcapaciteit van de bus.
        - vertrektijd: Eerste vertrektijd van de dienst.
        - eindtijd: Laatste eindtijd van de dienst.
    Output: Finale batterijpercentage na de simulatie.
    """
    battery = actual_capacity * 0.9  # Begin met 90% batterij
    min_battery = actual_capacity * 0.1  # Minimum batterijpercentage
    max_battery_day = actual_capacity * 0.9  # Maximaal 90% overdag
    max_battery_night = actual_capacity  # Maximaal 100% 's nachts
    charging_time_in_min = 15  # Minimaal 15 minuten opladen

    for i, row in circuit_planning.iterrows():
        # Converteer start en eindtijden naar datetime
        starting_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
        end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')

        # Controleer of de rit een dienst of materiaalrit is
        if row['activiteit'] in ['dienst rit', 'materiaal rit']:
            consumption = row['energieverbruik']
            battery -= consumption

        # Controleer of de bus idle is en genoeg tijd heeft om op te laden
        if row['activiteit'] == 'idle':
            idle_starting_time = datetime.strptime(row['starttijd'], '%H:%M:%S')
            idle_end_time = datetime.strptime(row['eindtijd'], '%H:%M:%S')
            
            idle_time = (idle_end_time - idle_starting_time).total_seconds() / 60  # Idle tijd in minuten

            # Laad de bus op als idle tijd minimaal 15 minuten is
            if idle_time >= charging_time_in_min:
                battery = charging(battery, actual_capacity, idle_starting_time, starting_time, end_time)

        # Check batterijstatus na elke stap
        if battery < min_battery:
            print(f"Waarschuwing: batterij te laag op {row['starttijd']}.")
        elif battery > max_battery_day and row['activiteit'] != 'idle':
            print(f"Waarschuwing: batterij boven 90% tijdens dienst op {row['starttijd']}.")

    return battery

# Voorbeeld data
actual_capacity = 285  # Capaciteit van de bus
starting_time = datetime.strptime('06:00', '%H:%M')
end_time = datetime.strptime('00:00', '%H:%M')

# Voer de simulatie uit
final_battery = simulate_battery(circuit_planning, actual_capacity, starting_time, end_time)
print(f"Uiteindelijke batterijstatus: {final_battery:.2f} kWh")