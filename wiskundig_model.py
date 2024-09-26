# ons wiskundige model 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

distance_matrix = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name = "Afstandsmatrix" )
all_sheets = pd.read_excel("Connexxion data - 2024-2025.xlsx", sheet_name = None)
print("Beschikbare sheets in het bestand:", list(all_sheets.keys()))

# we hebben 2 tabellen dus die kunnen we zo laten zien:
schedule = all_sheets['Dienstregeling']
distance_matrix = all_sheets['Afstandsmatrix']

# Parameters
max_capacity = 300 # maximale capaciteit in kWH
SOH = [85, 95] # State of Health
charging_speed_90 = 450 / 60 # kwh per minuut bij opladen tot 90%
charging_time_10 = 60 /60 # kwh per minuut bij opladen van 90% tot 100%
actual_capacity_90 = max_capacity * 0.9
actual_capacity = actual_capacity_90 
daytime_limit = actual_capacity_90 *0.9
consumption_per_km = [0.7, 2.5] # kWh per km

distance_matrix["afstand in meters"] = distance_matrix["afstand in meters"]/1000 # Ik bereken hier de afstand in km
distance_matrix = distance_matrix.rename(columns={'afstand in meters': 'afstand in km'}) # Hier hernoem ik de kolom.
distance_matrix["buslijn"] = distance_matrix["buslijn"].fillna("materiaalrit") # Ik wil laten zien dat alle waarde die NaN zijn materiaalritten zijn.

# Reistijden naar uren omrekenen
distance_matrix["min reistijd in min"] = distance_matrix["min reistijd in min"]/60 #max reistijd per uur.
distance_matrix["max reistijd in min"] = distance_matrix["max reistijd in min"]/60 #min reistijd per uur.
distance_matrix = distance_matrix.rename(columns={'min reistijd in min': 'min reistijd in uur'})
distance_matrix = distance_matrix.rename(columns={'max reistijd in min': 'max reistijd in uur'})

# we willen weten hoeveel km per uur een bus gemiddeld gaat.
distance_matrix["max_speed"] = distance_matrix["afstand in km"]/ distance_matrix['min reistijd in uur'] 
distance_matrix["min_speed"] = distance_matrix["afstand in km"]/ distance_matrix['max reistijd in uur'] 

# we willen weten hoeveel kwh je nodig hebt per rit: Verbruik per km =  0.7-2.5 kWh 
distance_matrix["max_energy"] = distance_matrix["afstand in km"]* 2.5
distance_matrix["min_energy"] = distance_matrix["afstand in km"]* 0.7

schedule['vertrektijd_dt'] = schedule['vertrektijd'].apply(lambda x: datetime.strptime(x, '%H:%M')) # vertrijd omzetten naar datetime
def calculate_end_time(row):
    """ telt de maximale reistijd op bij de vertrektijd, zodat er een kolom komt met eindtijd
    Parameters: row
    Output: eindtijd in HH:MM
    """
    travel_time = distance_matrix[(distance_matrix['startlocatie'] == row['startlocatie']) & 
                              (distance_matrix['eindlocatie'] == row['eindlocatie'])]['min reistijd in uur'].values
    if len(travel_time) > 0:
        travel_time_in_min = travel_time[0] # Converteer reistijd naar minuten
        end_time = row['vertrektijd_dt'] + timedelta(minutes=travel_time_in_min)
        return end_time
    else:
        return None 
schedule['eindtijd'] = schedule.apply(calculate_end_time, axis=1)

def charging(battery, actual_capacity, current_time, starting_time, end_time):
    """
    Beheren batterijstatus met verschillende regels voor opladen.
    Parameters: 
        - battery: huidige batterijpercentage in kWh
        - Dcap: batterijcapaciteit
        - huidige_tijd: het moment waarop de bus moet opladen
        - vertrektijd: tijd van de eerste busrit
        - eindtijd: tijd van de laatste busrit
    Output: Nieuwe batterij percentage in kWh
    """
    # Minimale batterijpercentage
    min_battery = 0.10 * actual_capacity  # De batterij mag niet onder dit percentage komen
    max_battery_day = 0.90 * actual_capacity  # Maximaal 90% overdag
    max_battery_night = actual_capacity  # Maximaal 100% na dienstregeling
    charging_time_in_min = 15  # Minimaal 15 minuten opladen
    charging_per_min = charging_speed_90  # Oplaadsnelheid tot 90%

    # Oplaadlimiet afhankelijk van het moment van de dag
    if current_time < starting_time or current_time > end_time:
        max_battery = max_battery_night
    else:
        max_battery = max_battery_day

    # Opladen gedurende de idle tijd (minimaal 15 minuten)
    charged_energy = charging_time_in_min * charging_per_min  # Energie in kWh

    if battery <= min_battery: 
        new_battery = battery + charged_energy
        if new_battery > max_battery:  # Zorgen dat het niet boven max uitkomt
            new_battery = max_battery
    else:
        new_battery = battery

    return new_battery

def battery_consumption(distance, current_time, starting_time, end_time):
    """
    Bereken het energieverbruik per rit op basis van afstand en snelheid.
    Houdt rekening met opladen voor en na de dienstregeling.
    Parameters: 
        - distance: afstand van de rit in km
        - huidige_tijd: huidige tijd van de busrit
        - vertrektijd: tijd van de eerste rit.
        - eindtijd: tijd van de laatste rit
        - bus_type: SOH van de bus (85% of 95%)
    Output: Nieuwe batterijpercentage na de rit en eventuele oplaadbeurt.
    """
    # Selecteer de juiste batterijcapaciteit
    battery_capacity = actual_capacity_90 * 0.9  # gemiddelde battert_capacity

    # Verbruik van de bus tijdens de rit (per km)
    battery_consumption = distance * np.mean(consumption_per_km)  # Verbruik in kWh
    remaining_battery = battery_capacity - battery_consumption

    # Checken of de bus moet opladen na de rit
    remaining_battery = charging(remaining_battery, battery_capacity, current_time, starting_time, end_time)
    
    return remaining_battery
 
""""   
Overdag niet meer dan 90% opladen = 229,5 - 256,5 kWh -> staat in functie opladen 
Altijd tenminste 15 min achtereen worden opgeladen. -> staat ook in opladen
Veiligheidsmarge van ongeveer 10% van de SOH -> bus minimaal SOC (state of charge) waarde van 10% 
Veiligheidsmarge van ongeveer 10% van de SOH -> bus minimaal SOC (state of charge) waarde van 10%. 
Minimale hoeveelheid in accu = 25,5 -28,5 kWh -> staat ook in opladen
Verbruik per km =  0.7-2.5 kWh 
Idling verbruikt het 0.01 kWh (verbruik van bus in stilstand) 
Lijn 401 eerste rit vertrekt om 6:04 vanuit de airport (Apt) naar Eindhoven Centraal (bst). 
401 laatste rit van Apt naar bst is om 00:31. Deze lijn zou klaar moeten zijn tussen de 00:53 en 00:56
Lijn 401 eerste rit vertrekt om 5:07 vanuit Eindhoven Centraal (bst) naar airport (Apt). 
401 laatste rit van bst naar apt is om 00:09. Deze lijn zou klaar moeten zijn tussen de 00:31 en 00:33. 
Lijn 400 eerste rit vertrekt om 7:19 vanuit de airport (Apt) naar Eindhoven Centraal (bst). 
400 laatste rit van Apt naar bst is om 20:46. Deze lijn zou klaar moeten zijn tussen de 21:07 en 21:09 
Lijn 400 eerste rit vertrekt om 06:52 vanuit Eindhoven Centraal (bst) naar airport (Apt). 
401 laatste rit van bst naar apt is om 19:37. Deze lijn zou klaar moeten zijn tussen de 19:58 en 20:00.  
"""