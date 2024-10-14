import pandas as pd

from datetime import datetime, timedelta
import numpy as np

# Load data
circuit_planning = pd.read_excel('omloopplanning.xlsx')
distance_matrix = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Afstandsmatrix")
schedule = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name="Dienstregeling")

# Parameters
max_capacity = 300  # maximale capaciteit in kWh
SOH = [85, 95]  # State of Health percentages
actual_capacity_90 = max_capacity * 0.9
actual_capacity = actual_capacity_90
charging_speed_90 = 450 / 60  # kWh per minuut voor opladen tot 90%
charging_speed_100 = 60 / 60  # kWh per minuut voor opladen van 90% tot 100%
consumption_per_km = (0.7 + 2.5) / 2  # kWh per kilometer
min_idle_time = 15  # in minuten
min_battery_percentage = 0.1  # minimaal 10% batterij vereist

# Data Preparation
distance_matrix["afstand in km"] = distance_matrix["afstand in meters"] / 1000
distance_matrix["min reistijd in uur"] = distance_matrix["min reistijd in min"] / 60
distance_matrix["max reistijd in uur"] = distance_matrix["max reistijd in min"] / 60
distance_matrix["buslijn"] = distance_matrix["buslijn"].fillna("materiaalrit")
distance_matrix["max_energy"] = distance_matrix["afstand in km"] * 2.5
distance_matrix["min_energy"] = distance_matrix["afstand in km"] * 0.7

# Bereken eindtijd van elke rit
schedule['vertrektijd_dt'] = schedule['vertrektijd'].apply(lambda x: datetime.strptime(x, '%H:%M'))

def calculate_end_time(row):
    travel_time = distance_matrix[(distance_matrix['startlocatie'] == row['startlocatie']) &
                                  (distance_matrix['eindlocatie'] == row['eindlocatie'])]['min reistijd in uur'].values
    if len(travel_time) > 0:
        travel_time_in_min = travel_time[0] * 60
        end_time = row['vertrektijd_dt'] + timedelta(minutes=travel_time_in_min)
        return end_time
    else:
        return None

schedule['eindtijd'] = schedule.apply(calculate_end_time, axis=1)

# Functie om ritten toe te voegen aan het schema
bussen = {
    'omloop': [],
    'busnummer': [],
    'startlocatie': [],
    'eindlocatie': [],
    'starttijd': [],
    'eindtijd': [],
    'activiteit': [],
    'buslijn': [],
    'energieverbruik': []
}

def add_ride_to_bus(busnummer, startlocatie, eindlocatie, starttijd, eindtijd, activiteit, buslijn, energieverbruik):
    bussen['omloop'].append(busnummer)
    bussen['startlocatie'].append(startlocatie)
    bussen['eindlocatie'].append(eindlocatie)
    bussen['starttijd'].append(starttijd.strftime('%H:%M:%S'))
    bussen['eindtijd'].append(eindtijd.strftime('%H:%M:%S'))
    bussen['activiteit'].append(activiteit)
    bussen['buslijn'].append(buslijn)
    bussen['energieverbruik'].append(energieverbruik)

# Simulatie van batterijverbruik en opladen
def simulate_battery(busnummer, bus_rides, actual_capacity):
    battery = actual_capacity * 0.9  # Start de dag met 90% batterij
    min_battery = actual_capacity * 0.1  # Minimaal 10% batterij
    max_battery_day = actual_capacity * 0.9
    max_battery_night = actual_capacity

    for i, row in bus_rides.iterrows():
        starttijd = datetime.strptime(row['starttijd'], '%H:%M:%S')
        eindtijd = datetime.strptime(row['eindtijd'], '%H:%M:%S')
        consumption = row['energieverbruik']
        battery -= consumption  # Verminder de batterij op basis van energieverbruik

        # Controleer of batterij laag is, en plan oplaadmoment
        if battery < min_battery:
            print(f"Bus {busnummer} battery too low at {row['starttijd']}. Planning recharge.")
            idle_start_time = starttijd
            idle_end_time = eindtijd
            idle_time = (idle_end_time - idle_start_time).total_seconds() / 60
            if idle_time >= min_idle_time:
                battery = charging(battery, actual_capacity, idle_start_time, starttijd, eindtijd)
            else:
                print(f"Charging time too short, only {idle_time} minutes available.")

        add_ride_to_bus(busnummer, row['startlocatie'], row['eindlocatie'], starttijd, eindtijd, row['activiteit'], row['buslijn'], row['energieverbruik'])

    return battery

# Oplaadfunctie
def charging(battery, actual_capacity, current_time, start_time, end_time):
    if current_time < start_time or current_time > end_time:
        max_battery = actual_capacity
    else:
        max_battery = actual_capacity * 0.9

    charged_energy = min_idle_time * charging_speed_90
    new_battery = battery + charged_energy if battery < max_battery else battery
    return min(new_battery, max_battery)

# Controle van routecontinuïteit
def check_route_continuity(circuit_planning):
    for i in range(len(circuit_planning) - 1):
        current_end_location = circuit_planning.iloc[i]['eindlocatie']
        next_start_location = circuit_planning.iloc[i + 1]['startlocatie']
        if current_end_location != next_start_location:
            print(f"Warning: Route continuity issue between {circuit_planning.iloc[i]['omloop nummer']:.0f} ending at {current_end_location} and next route starting at {next_start_location}.")
            return False
    return True

# Loopen door busplanning en ritten inplannen
for busnummer in circuit_planning['omloop nummer'].unique():
    bus_rides = circuit_planning[circuit_planning['omloop nummer'] == busnummer]
    final_battery = simulate_battery(busnummer, bus_rides, actual_capacity)

# Resultaat opslaan in een DataFrame
final_schedule = pd.DataFrame(bussen)

# Laat het schema zien
print(final_schedule)