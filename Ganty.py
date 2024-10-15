import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
uploaded_file = pd.read_excel('omloopplanning.xlsx')
distance_matrix = pd.read_excel('Connexxion data - 2024-2025.xlsx ',sheet_name = "Afstandsmatrix")
time_table = pd.read_excel('Connexxion data - 2024-2025.xlsx ',sheet_name = "Dienstregeling")
uploaded_file.head()
def plot_schedule_from_excel(uploaded_file):
    """Plot een Gantt-grafiek voor busplanning op basis van een Excel-bestand."""
    
    # Zorg ervoor dat de juiste datatypes zijn ingesteld
    uploaded_file['starttijd'] = pd.to_datetime(uploaded_file['starttijd'])
    uploaded_file['eindtijd'] = pd.to_datetime(uploaded_file['eindtijd'])
    
    # Bereken de duur in uren
    uploaded_file['duration'] = (uploaded_file['eindtijd'] - uploaded_file['starttijd']).dt.total_seconds() / 3600

    # Kleurmap voor verschillende buslijnen
    color_map = {
        '400.0': 'blue',
        '401.0': 'yellow'
    }

    # Zet de buslijnwaarden om naar strings
    uploaded_file['buslijn'] = uploaded_file['buslijn'].astype(str)

    # Voeg een nieuwe kolom toe met de kleur op basis van de buslijn
    uploaded_file['color'] = uploaded_file['buslijn'].map(color_map).fillna('gray')

    # Maak een figuur voor het plotten
    fig, ax = plt.subplots(figsize=(12, 6))

    # Omloopnummers op de Y-as
    omloopnummers = uploaded_file['omloop nummer'].unique()
    omloop_indices = {omloop: i for i, omloop in enumerate(omloopnummers)}

    # Loop door de unieke omloopnummers
    for omloop in omloopnummers:
        trips = uploaded_file[uploaded_file['omloop nummer'] == omloop]
        
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
                    color=color, edgecolor='black', label=trip['buslijn'] if trip['buslijn'] not in ax.get_legend_handles_labels()[1] else "")
    
    # Zet de Y-ticks en labels voor de omloopnummers
    ax.set_yticks(list(omloop_indices.values()))
    ax.set_yticklabels(list(omloop_indices.keys()))

    # Set axis labels and title
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Omloopnummer')
    ax.set_title('Gantt Chart for Bus Scheduling')

    # Voeg een legenda toe (voorkom dubbele labels)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), title='Buslijnen')

    plt.show()

# Voorbeeld van aanroepen van de functie (upload je DataFrame)
plot_schedule_from_excel(uploaded_file)
